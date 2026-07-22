// ndt_d2d_mapper_node.cpp
//
// Nodo de REGISTRO + MAPEO scan-to-map usando NDT D2D (Distribution-to-Distribution).
//
// Este nodo es ahora el DUEÑO del mapa (Bonxai::VoxelGrid<NDTCell> interno,
// persistente entre callbacks). Ya no depende de un tópico externo de mapa:
// puedes cambiar el proveedor de scans (el productor de NDTMap) sin tocar
// este nodo.
//
// Flujo por cada scan recibido (celdas en FRAME LOCAL del sensor):
//   1. Si el mapa está vacío -> se usa la pose del propio scan (sensor->mundo)
//      para transformar sus celdas e inicializar el mapa (bootstrap).
//   2. Si el mapa ya existe -> se registra el scan contra el mapa acumulado
//      (D2D NDT), partiendo como 'guess' de la pose del sensor que trae el
//      propio mensaje. Se obtiene una pose corregida.
//   3. Las celdas del scan se transforman al frame del mapa con la pose
//      corregida y se FUSIONAN (merge exacto de gaussianas) en el grid.
//   4. El mapa completo se publica por timer (no en cada scan) para no
//      saturar la red.
//
// NOTA IMPORTANTE: asumo que tu mensaje NDTMap de entrada (el scan) trae,
// además de las celdas, la pose del sensor respecto al mundo, ya que
// mencionas "la voxelización se hace guardando esta pose". Si el campo no
// se llama 'pose' en tu .msg real, ajusta 'extractSensorPose()'.

#include <rclcpp/rclcpp.hpp>

#include <std_srvs/srv/empty.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>

#include <fstream>
#include <filesystem>

#include "gicp_mapper/msg/ndt_cloud.hpp"
#include "gicp_mapper/msg/ndt_cell.hpp"
#include "ndt_cell.hpp"


#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <bonxai.hpp>    // AJUSTA include real de Bonxai
#include <omp.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>



using NDTCell = gicp_mapper::NDTCell;
using NDTGrid   = Bonxai::VoxelGrid<NDTCell>;
using NDTMapMsg = gicp_mapper::msg::NDTCloud;     
using NDTMapPtr = NDTMapMsg::SharedPtr;

using EmptySrv = std_srvs::srv::Empty;

enum class RegistrationMetric { MAHALANOBIS, KLD };

struct PlyPoint
{
    Eigen::Vector3d voxel_center; //Centro voxel

    uint8_t r;      
    uint8_t g;
    uint8_t b;

    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;

    uint32_t num_points;
};

// ---------------------------------------------------------------------

static inline double symmetricKLD(const Eigen::Vector3d & mu0, const Eigen::Matrix3d & Sigma0,
                                   const Eigen::Vector3d & mu1, const Eigen::Matrix3d & Sigma1,
                                   double reg)
{
  Eigen::Matrix3d S0 = Sigma0 + reg * Eigen::Matrix3d::Identity();
  Eigen::Matrix3d S1 = Sigma1 + reg * Eigen::Matrix3d::Identity();

  Eigen::LDLT<Eigen::Matrix3d> ldlt0(S0), ldlt1(S1);
  if (ldlt0.info() != Eigen::Success || ldlt1.info() != Eigen::Success)
    return std::numeric_limits<double>::max();

  Eigen::Matrix3d Info0 = S0.inverse();
  Eigen::Matrix3d Info1 = S1.inverse();

  const double logdet0 = std::log(std::max(S0.determinant(), 1e-300));
  const double logdet1 = std::log(std::max(S1.determinant(), 1e-300));

  const Eigen::Vector3d d = mu1 - mu0;
  constexpr int k = 3;

  // KL(P0||P1)
  const double kl_01 = 0.5 * ( (Info1 * S0).trace() + d.transpose() * Info1 * d - k + (logdet1 - logdet0) );
  // KL(P1||P0)
  const double kl_10 = 0.5 * ( (Info0 * S1).trace() + d.transpose() * Info0 * d - k + (logdet0 - logdet1) );

  return kl_01 + kl_10;
}





static inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d S;
  S <<     0, -v.z(),  v.y(),
        v.z(),      0, -v.x(),
       -v.y(),  v.x(),      0;
  return S;
}

static inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi)
{
  const double theta = phi.norm();
  if (theta < 1e-12) return Eigen::Matrix3d::Identity() + skew(phi);
  return Eigen::AngleAxisd(theta, phi / theta).toRotationMatrix();
}

static Eigen::Isometry3d poseToIso(const geometry_msgs::msg::PoseWithCovariance & pwc)
{
  const auto & p = pwc.pose;
  Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  q.normalize();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  return T;
}

// static void isoToPose(const Eigen::Isometry3d & T, geometry_msgs::msg::PoseWithCovariance & pwc)
// {
//   const Eigen::Quaterniond q(T.rotation());
//   pwc.pose.position.x = T.translation().x();
//   pwc.pose.position.y = T.translation().y();
//   pwc.pose.position.z = T.translation().z();
//   pwc.pose.orientation.x = q.x();
//   pwc.pose.orientation.y = q.y();
//   pwc.pose.orientation.z = q.z();
//   pwc.pose.orientation.w = q.w();
// }

// Convierte las celdas del msg (en FRAME LOCAL del sensor) a NDTCell, tal cual,
// sin transformar todavía.
static std::vector<NDTCell> fromMsgLocal(const NDTMapMsg & msg)
{
  std::vector<NDTCell> cells;
  cells.reserve(msg.cells.size());
  for (const auto & c : msg.cells) {
    NDTCell cell;
    cell.sum = Eigen::Vector3d(c.sum.x, c.sum.y, c.sum.z);
    cell.sum_outer << c.sum_outer[0], c.sum_outer[1], c.sum_outer[2],
                      c.sum_outer[3], c.sum_outer[4], c.sum_outer[5],
                      c.sum_outer[6], c.sum_outer[7], c.sum_outer[8];
    cell.mean = Eigen::Vector3d(c.mean_x, c.mean_y, c.mean_z);
    cell.covariance << c.covariance[0], c.covariance[1], c.covariance[2],
                       c.covariance[3], c.covariance[4], c.covariance[5],
                       c.covariance[6], c.covariance[7], c.covariance[8];
    cell.num_points = c.num_points;

    cell.r = c.b;
    cell.g = c.g;
    cell.b = c.r;

    cell.valid = c.valid;
    cells.push_back(cell);
  }
  return cells;
}

// Transforma una celda NDT (sum, sum_outer, mean, covariance) por una
// isometría T = [R | t]. sum_outer y covariance rotan como formas cuadráticas;
// sum y mean como vectores; num_points no cambia.
//
// sum_outer_new = R * sum_outer * R^T + R*sum*t^T + t*sum^T*R^T + n*t*t^T
// (viene de expandir sum_outer = sum_i (p_i)(p_i)^T con p_i' = R*p_i + t)
static NDTCell transformCell(const NDTCell & c, const Eigen::Isometry3d & T)
{
  NDTCell out = c;
  if (!c.valid) return out;

  const Eigen::Matrix3d R = T.rotation();
  const Eigen::Vector3d t = T.translation();
  const double n = static_cast<double>(c.num_points);

  out.mean = R * c.mean + t;
  out.sum  = R * c.sum + n * t;

  const Eigen::Matrix3d Rs  = R * c.sum_outer * R.transpose();
  const Eigen::Vector3d Rst = R * c.sum;              // R * sum (vector)
  out.sum_outer = Rs
                + Rst * t.transpose()
                + t * Rst.transpose()
                + n * (t * t.transpose());

  out.covariance = R * c.covariance * R.transpose();
  return out;
}

// Fusión exacta de dos celdas gaussianas (misma celda de voxel, misma región).
// static NDTCell mergeCell(const NDTCell & a, const NDTCell & b)
// {
//   if (!a.valid) return b;
//   if (!b.valid) return a;

//   NDTCell out;
//   out.sum        = a.sum + b.sum;
//   out.sum_outer  = a.sum_outer + b.sum_outer;
//   out.num_points = a.num_points + b.num_points;
//   out.valid      = true;

//   const double n = static_cast<double>(out.num_points);
//   out.mean = out.sum / n;

//   // covarianza muestral: E[xx^T] - mean*mean^T, con corrección de Bessel
//   const Eigen::Matrix3d Exx = out.sum_outer / n;
//   Eigen::Matrix3d cov = Exx - out.mean * out.mean.transpose();
//   if (n > 1.0) cov *= n / (n - 1.0);
//   out.covariance = cov;

//   return out;
// }


NDTCell mergeCell(const NDTCell & a, const NDTCell & b)
{
  if (!a.valid) return b;
  if (!b.valid) return a;

  NDTCell out;
  out.sum        = a.sum + b.sum;
  out.sum_outer  = a.sum_outer + b.sum_outer;
  out.num_points = a.num_points + b.num_points;
  out.valid      = true;

  const double n = static_cast<double>(out.num_points);
  out.mean = out.sum / n;

  const Eigen::Matrix3d Exx = out.sum_outer / n;
  Eigen::Matrix3d cov = Exx - out.mean * out.mean.transpose();
  if (n > 1.0) cov *= n / (n - 1.0);
  out.covariance = cov;

  // Fusión ponderada del color
  const double na = static_cast<double>(a.num_points);
  const double nb = static_cast<double>(b.num_points);
  out.r = static_cast<uint8_t>(std::round((a.r * na + b.r * nb) / n));
  out.g = static_cast<uint8_t>(std::round((a.g * na + b.g * nb) / n));
  out.b = static_cast<uint8_t>(std::round((a.b * na + b.b * nb) / n));

  return out;
}

// ---------------------------------------------------------------------
//  Matcher D2D (paralelizado) — igual que antes, pero ahora opera sobre
//  el grid interno persistente del nodo (pasado por referencia).
// ---------------------------------------------------------------------
class NDTD2DMatcher
{
public:
  struct Params
  {
    int max_iterations = 30;
    double translation_eps = 1e-4;
    double rotation_eps = 1e-4;
    double max_mahalanobis_dist = 9.0;
    double covariance_reg = 1e-6;
    int min_points_per_cell = 5;
    RegistrationMetric metric = RegistrationMetric::MAHALANOBIS;   // NUEVO
    double max_kld = 50.0;   // NUEVO: umbral de rechazo por correspondencia para KLD
  };

    NDTD2DMatcher(){}

  explicit NDTD2DMatcher(const Params & p) : params_(p) {}

  double computeScore(NDTGrid & target_grid,const std::vector<NDTCell>& source_cells_local,const Eigen::Isometry3d& T,int& correspondences)
    {
    double score = 0.0;
    correspondences = 0;

    auto accessor = target_grid.createAccessor();

    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d t = T.translation();


    for(const auto& sc : source_cells_local)
    {
        if(!sc.valid || 
          sc.num_points < static_cast<uint32_t>(params_.min_points_per_cell))
            continue;

        Eigen::Vector3d mean_world = R * sc.mean + t;

        auto coord = target_grid.posToCoord(
            mean_world.x(),
            mean_world.y(),
            mean_world.z()
        );


        const NDTCell* tc = accessor.value(coord,false);

        if(tc == nullptr || !tc->valid)
            continue;



        Eigen::Matrix3d cov_s = R * sc.covariance * R.transpose();


        if (params_.metric == RegistrationMetric::MAHALANOBIS) {
            Eigen::Matrix3d Sigma = tc->covariance + cov_s + params_.covariance_reg * Eigen::Matrix3d::Identity();
            Eigen::Matrix3d Info = Sigma.inverse();
            Eigen::Vector3d r = mean_world - tc->mean;
            double mdist = r.transpose() * Info * r;
            if (mdist > params_.max_mahalanobis_dist) continue;
            score += mdist;
          } else { // KLD
            double kld = symmetricKLD(mean_world, cov_s, tc->mean, tc->covariance, params_.covariance_reg);
            if (kld > params_.max_kld) continue;
            score += kld;
          }
          correspondences++;
          }
        
        
          if(correspondences == 0)
              return std::numeric_limits<double>::max();
        

          return score;
        }

  bool match(NDTGrid & target_grid,
             const std::vector<NDTCell> & source_cells_local,
             const Eigen::Isometry3d & T_init,
             Eigen::Isometry3d & T_out,
             double & final_score,
             int & used_correspondences_out)
  {
    Eigen::Matrix3d R = T_init.rotation();
    Eigen::Vector3d t = T_init.translation();

    bool converged = false;
    double score = 0.0;
    int used_correspondences = 0;
    const int num_threads = omp_get_max_threads();
    const int n = static_cast<int>(source_cells_local.size());

    for (int iter = 0; iter < params_.max_iterations; ++iter) {

      std::vector<Eigen::Matrix<double,6,6>> H_local(num_threads, Eigen::Matrix<double,6,6>::Zero());
      std::vector<Eigen::Matrix<double,6,1>> b_local(num_threads, Eigen::Matrix<double,6,1>::Zero());
      std::vector<int> count_local(num_threads, 0);
      std::vector<double> score_local(num_threads, 0.0);

      #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        auto local_accessor = target_grid.createAccessor(); // accessor propio por hilo

        Eigen::Matrix<double,6,6> H_t = Eigen::Matrix<double,6,6>::Zero();
        Eigen::Matrix<double,6,1> b_t = Eigen::Matrix<double,6,1>::Zero();
        int count_t = 0;
        double score_t = 0.0;

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
          const NDTCell & sc = source_cells_local[i];
          if (!sc.valid || sc.num_points < static_cast<uint32_t>(params_.min_points_per_cell)) continue;

          const Eigen::Vector3d mean_s_world = R * sc.mean + t;
          const auto coord = target_grid.posToCoord(mean_s_world.x(), mean_s_world.y(), mean_s_world.z());
          const NDTCell * tc = local_accessor.value(coord, false);
          if (tc == nullptr || !tc->valid) continue;

          const Eigen::Matrix3d cov_s_world = R * sc.covariance * R.transpose();
          Eigen::Matrix3d Sigma = tc->covariance + cov_s_world;
          Sigma += params_.covariance_reg * Eigen::Matrix3d::Identity();

          Eigen::Matrix3d Info;
          {
            Eigen::LDLT<Eigen::Matrix3d> ldlt(Sigma);
            if (ldlt.info() != Eigen::Success) continue;
            Info = Sigma.inverse();
          }
          double residual_metric = 0.0;
          const Eigen::Vector3d r = mean_s_world - tc->mean;

          if (params_.metric == RegistrationMetric::MAHALANOBIS) {
            Eigen::Matrix3d Sigma = tc->covariance + cov_s_world + params_.covariance_reg * Eigen::Matrix3d::Identity();
            Eigen::LDLT<Eigen::Matrix3d> ldlt(Sigma);
            if (ldlt.info() != Eigen::Success) continue;
            Info = Sigma.inverse();
            residual_metric = r.transpose() * Info * r;
            if (residual_metric > params_.max_mahalanobis_dist) continue;
          } else { // KLD: gradiente de medias con Sigma1^-1 (parte "viva" de la traza se ignora en el Jacobiano)
            Eigen::Matrix3d Sigma1 = tc->covariance + params_.covariance_reg * Eigen::Matrix3d::Identity();
            Eigen::LDLT<Eigen::Matrix3d> ldlt(Sigma1);
            if (ldlt.info() != Eigen::Success) continue;
            Info = Sigma1.inverse();
            double kld = symmetricKLD(mean_s_world, cov_s_world, tc->mean, tc->covariance, params_.covariance_reg);
            if (kld > params_.max_kld) continue;
            residual_metric = kld; // solo para el score acumulado, no afecta H/b
          }

score_t += residual_metric;
++count_t;

Eigen::Matrix<double,3,6> J;
J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
J.block<3,3>(0,3) = -skew(R * sc.mean);

H_t += J.transpose() * Info * J;
b_t += J.transpose() * Info * r;

          // const double mdist = r.transpose() * Info * r;
          // if (mdist > params_.max_mahalanobis_dist) continue;

          // score_t += mdist;
          // ++count_t;

          // Eigen::Matrix<double,3,6> J;
          // J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
          // J.block<3,3>(0,3) = -skew(R * sc.mean);

          // H_t += J.transpose() * Info * J;
          // b_t += J.transpose() * Info * r;
        }

        H_local[tid] = H_t;
        b_local[tid] = b_t;
        count_local[tid] = count_t;
        score_local[tid] = score_t;
      }

      Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
      Eigen::Matrix<double,6,1> b = Eigen::Matrix<double,6,1>::Zero();
      used_correspondences = 0;
      score = 0.0;
      for (int k = 0; k < num_threads; ++k) {
        H += H_local[k];
        b += b_local[k];
        used_correspondences += count_local[k];
        score += score_local[k];
      }

      if (used_correspondences < 6) return false;

      H.diagonal().array() += 1e-6;
      Eigen::Matrix<double,6,1> dx = H.ldlt().solve(-b);

      t += dx.head<3>();
      R = expSO3(dx.tail<3>()) * R;

      Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
      R = svd.matrixU() * svd.matrixV().transpose();

      if (dx.head<3>().norm() < params_.translation_eps &&
          dx.tail<3>().norm() < params_.rotation_eps) {
        converged = true;
        break;
      }
    }

    T_out = Eigen::Isometry3d::Identity();
    T_out.linear() = R;
    T_out.translation() = t;
    final_score = score;
    used_correspondences_out = used_correspondences;
    return converged;
  }

private:
  Params params_;
};

// ---------------------------------------------------------------------
//  Nodo ROS2 — mapa acumulado internamente (map owner)
// ---------------------------------------------------------------------
class NDTD2DMapperNode : public rclcpp::Node
{
public:
  NDTD2DMapperNode() : Node("ndt_d2d_mapper_node")
  {
    declare_parameter("scan_topic",       "ndt_cloud");             // NDTMap del scan (frame local)

    declare_parameter("resolution",   0.02);                    // resolucion del VoxelGrid del mapa

    declare_parameter("max_iterations",       64);
    declare_parameter("translation_eps",      1e-4);
    declare_parameter("rotation_eps",         1e-4);
    declare_parameter("max_mahalanobis_dist", 9.0);
    declare_parameter("covariance_reg",       1e-6);
    declare_parameter("min_points_per_cell",  5);
    declare_parameter("min_correspondences",  6);
    declare_parameter("fitness_reject",       1e6);
    declare_parameter("drop_on_reject",       false);

    declare_parameter("use_ground_truth", false);

    declare_parameter("registration_metric", "mahalanobis");
    declare_parameter("max_kld", 50.0);

    NDTD2DMatcher::Params mp;
    mp.max_iterations       = static_cast<int>(get_parameter("max_iterations").as_int());
    mp.translation_eps      = get_parameter("translation_eps").as_double();
    mp.rotation_eps         = get_parameter("rotation_eps").as_double();
    mp.max_mahalanobis_dist = get_parameter("max_mahalanobis_dist").as_double();
    mp.covariance_reg       = get_parameter("covariance_reg").as_double();
    mp.min_points_per_cell  = static_cast<int>(get_parameter("min_points_per_cell").as_int());

    std::string metric_str = get_parameter("registration_metric").as_string();
    mp.max_kld = get_parameter("max_kld").as_double();

    if (metric_str == "mahalanobis") {
      mp.metric = RegistrationMetric::MAHALANOBIS;
    } else if (metric_str == "kld") {
      mp.metric = RegistrationMetric::KLD;
    } else {
      RCLCPP_WARN(get_logger(),
        "[NDT D2D Mapper] registration_metric='%s' desconocido, usando 'mahalanobis'",
        metric_str.c_str());
      mp.metric = RegistrationMetric::MAHALANOBIS;
    }


    matcher_ = std::make_unique<NDTD2DMatcher>(mp);




    map_resolution_       = get_parameter("resolution").as_double();
    min_correspondences_  = static_cast<int>(get_parameter("min_correspondences").as_int());
    fitness_reject_        = get_parameter("fitness_reject").as_double();
    drop_on_reject_         = get_parameter("drop_on_reject").as_bool();

    use_ground_truth         = get_parameter("use_ground_truth").as_bool();

    map_grid_ = std::make_shared<NDTGrid>(map_resolution_);

    // ── Único suscriptor: el scan ────────────────────────────────────────────

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10000));
    qos.reliable();

    scan_sub_ = create_subscription<NDTMapMsg>(
        get_parameter("scan_topic").as_string(), qos,
        [this](NDTMapPtr msg) { enqueue(std::move(msg)); });

    save_map_srv_ = create_service<EmptySrv>("~/save_map",std::bind(
        &NDTD2DMapperNode::saveMapPLY,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    )
);

    // ── Timer de publicación del mapa (no en cada scan) ─────────────────────

    running_       = true;
    worker_thread_ = std::thread(&NDTD2DMapperNode::workerLoop, this);

    RCLCPP_INFO(get_logger(),
      "[NDT D2D Mapper] scan_topic=%s  res=%.3f metrica=%s" ,
      get_parameter("scan_topic").as_string().c_str(),
      map_resolution_,
      metric_str.c_str());
  }

  ~NDTD2DMapperNode() override
  {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  static constexpr size_t kMaxQueue = 50;

  // ── Extrae la pose del sensor respecto al mundo desde el propio msg.
  // AJUSTA si tu campo no se llama 'pose' o no es PoseWithCovariance.
  static Eigen::Isometry3d extractSensorPose(const NDTMapMsg & msg)
  {
    return poseToIso(msg.pose); // 
  }

  // ── Cola con límite ────────────────────────────────────────────────────
  void enqueue(NDTMapPtr msg)
  {
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if (msg_queue_.size() >= kMaxQueue) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "[NDT D2D Mapper] Cola llena (%zu) — scan descartado", msg_queue_.size());
        return;
      }
      msg_queue_.push(std::move(msg));
    }
    queue_cv_.notify_one();
  }

  void workerLoop()
  {
    while (running_) {
      NDTMapPtr msg;
      {
        std::unique_lock<std::mutex> lk(queue_mutex_);
        queue_cv_.wait(lk, [this] { return !msg_queue_.empty() || !running_; });
        if (!running_ && msg_queue_.empty()) break;
        msg = std::move(msg_queue_.front());
        msg_queue_.pop();
      }
      try {
        processScan(msg);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "[NDT D2D Mapper] Excepcion en processScan: %s", e.what());
      } catch (...) {
        RCLCPP_ERROR(get_logger(), "[NDT D2D Mapper] Excepcion desconocida — scan descartado");
      }
    }
  }

  // ── Procesamiento de un scan: registra (si hay mapa) y fusiona ──────────
  void processScan(const NDTMapPtr & msg)
  {

    RCLCPP_INFO(get_logger(), "CALLBACK");

    const auto cells_local = fromMsgLocal(*msg);
    if (cells_local.empty()) {
      RCLCPP_WARN(get_logger(), "[NDT D2D Mapper] Scan sin celdas — ignorado");
      return;
    }

    const Eigen::Isometry3d sensor_pose = extractSensorPose(*msg); // guess inicial

    std::lock_guard<std::mutex> lock(map_mutex_);

    Eigen::Isometry3d corrected = sensor_pose;

    if (!map_has_data_) {
      // Bootstrap: no hay mapa todavía, se usa la pose del scan tal cual.
      RCLCPP_INFO(get_logger(), "[NDT D2D Mapper] Bootstrap: inicializando mapa con el primer scan");
    } else {
      Eigen::Isometry3d T_rel;
      double score_after = 0.0;
      int used_corr = 0;

      if(!use_ground_truth){  

      double score_before;
      int corr_before;

      score_before =
      matcher_->computeScore(
          *map_grid_,
          cells_local,
          sensor_pose,
          corr_before
      );

      const auto t0 = now();
      const bool converged = matcher_->match(*map_grid_, cells_local, sensor_pose, T_rel, score_after, used_corr);
      const double ms = (now() - t0).seconds() * 1e3;

      // const bool ok = converged
      //                 && used_corr >= min_correspondences_
      //                 && T_rel.matrix().allFinite()
      //                 && score <= fitness_reject_;

      // bool improved =
      //   converged &&
      //   used_corr >= min_correspondences_ &&
      //   score_after < score_before;


      bool improved = used_corr >= min_correspondences_ &&
        score_after < score_before;

      const bool ok = improved &&
        T_rel.matrix().allFinite();

      if (ok) {
        corrected    = T_rel;
        RCLCPP_INFO(get_logger(), "[NDT D2D Mapper] OK  Antes=%.4f  Despues=%.4f corr=%d  %.2f ms",
                    score_before, score_after, used_corr, ms);
      } else {
        RCLCPP_WARN(get_logger(),
          "[NDT D2D Mapper] Rechazado (Antes=%.4f  Despues=%.4f corr=%d)  %.2f ms — %s",
           score_before, score_after, used_corr, ms,
          drop_on_reject_ ? "descartado" : "se usa guess sin corregir para fusionar igualmente");
        if (drop_on_reject_) return;
        // corrected se queda como sensor_pose (guess), y se sigue fusionando
        // igualmente para no perder cobertura del mapa si así lo prefieres.
        // Si prefieres NO fusionar cuando el registro falla, añade "return;" aquí.
      }
    }
    }

    // ── Fusión: transformar celdas del scan al frame del mapa y hacer merge ──
    mergeIntoMap(cells_local, corrected);
    map_has_data_ = true;
  }

  // ── Fusión de celdas transformadas dentro del grid interno ───────────────
  void mergeIntoMap(const std::vector<NDTCell> & cells_local, const Eigen::Isometry3d & T)
  {
    auto accessor = map_grid_->createAccessor();

    for (const auto & c_local : cells_local) {
      if (!c_local.valid) continue;

      const NDTCell c_world = transformCell(c_local, T);

      const auto coord = map_grid_->posToCoord(c_world.mean.x(), c_world.mean.y(), c_world.mean.z());
      const NDTCell * existing = accessor.value(coord, false);

      const NDTCell merged = (existing == nullptr) ? c_world : mergeCell(*existing, c_world);
      accessor.setValue(coord, merged);
    }
  }

  void saveMapPLY(const std::shared_ptr<std_srvs::srv::Empty::Request>,std::shared_ptr<std_srvs::srv::Empty::Response>){

    std::string ply_filename = "voxeland_pointcloud.ply";
    std::ofstream outfile(ply_filename);

    if(!outfile.is_open())
    {
        RCLCPP_ERROR(
            get_logger(),
            "No se pudo abrir %s",
            ply_filename.c_str());

        return;
    }

    std::vector<PlyPoint> points;

    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        if(!map_has_data_)
        {
            RCLCPP_WARN(
                get_logger(),
                "Mapa vacío, no se guarda PLY");

            return;
        }

        map_grid_->forEachCell([&](NDTCell& cell, const Bonxai::CoordT& coord)
        {
            if (!cell.valid)
                return;

            auto pos = map_grid_->coordToPos(coord);

            PlyPoint p;

            p.voxel_center = Eigen::Vector3d(
                pos.x,
                pos.y,
                pos.z
            );

            p.r = cell.r;
            p.g = cell.g;
            p.b = cell.b;

            p.mean = cell.mean;
            p.covariance = cell.covariance;

            p.num_points = cell.num_points;

            points.push_back(p);
        
            // points.push_back({
            //     map_grid_->coordToPos(coord),
            //     cell.r,
            //     cell.g,
            //     cell.b,
            //     cell.mean,
            //     cell.covariance
            // });
        });
    }

    // -----------------------------
    // Cabecera PLY
    // -----------------------------

    outfile
    << "ply\n"
    << "format ascii 1.0\n"
    << "element vertex "
    << points.size()
    << "\n"
    << "property float x\n"
    << "property float y\n"
    << "property float z\n"
    << "property uchar red\n"
    << "property uchar green\n"
    << "property uchar blue\n"
    << "end_header\n";



    // -----------------------------
    // Datos
    // -----------------------------

    for(const auto& p : points)
    {
        outfile
        << p.voxel_center.x() << " "
        << p.voxel_center.y() << " "
        << p.voxel_center.z() << " "
        << static_cast<int>(p.r) << " "
        << static_cast<int>(p.g) << " "
        << static_cast<int>(p.b)
        << "\n";
    }
    outfile.close();



    RCLCPP_INFO(
        get_logger(),
        "Mapa guardado: %s (%zu puntos)",
        ply_filename.c_str(),
        points.size());



        std::string txt_filename = "voxeland_pointcloud_comp.txt";
        std::ofstream txtfile(txt_filename);

    if(!txtfile.is_open())
      {
          RCLCPP_ERROR(
              get_logger(),
              "No se pudo abrir %s",
              txt_filename.c_str());
          
          return;
      }


    txtfile << "# voxel_x voxel_y voxel_z "
        << "mean_x mean_y mean_z "
        << "cov_xx cov_xy cov_xz "
        << "cov_yx cov_yy cov_yz "
        << "cov_zx cov_zy cov_zz "
        << "num_points\n";


    for(const auto& p : points)
    {
        txtfile
            << p.voxel_center.x() << " "
            << p.voxel_center.y() << " "
            << p.voxel_center.z() << " "
    
            << p.mean.x() << " "
            << p.mean.y() << " "
            << p.mean.z() << " ";
    
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                txtfile << p.covariance(i,j) << " ";
            }
        }
      
        txtfile
            << p.num_points
            << "\n";
    }

    txtfile.close();
}

  // ── Miembros ─────────────────────────────────────────────────────────────
  rclcpp::Subscription<NDTMapMsg>::SharedPtr scan_sub_;

  std::unique_ptr<NDTD2DMatcher> matcher_;

  rclcpp::Service<EmptySrv>::SharedPtr save_map_srv_;

  // Mapa interno, persistente, propiedad de este nodo.
  std::shared_ptr<NDTGrid> map_grid_;
  bool        map_has_data_ = false;
  double      map_resolution_;
  std::mutex  map_mutex_;

  std::queue<NDTMapPtr>   msg_queue_;
  std::mutex              queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool>       running_{false};
  std::thread             worker_thread_;

  int    min_correspondences_;
  double fitness_reject_;
  bool   drop_on_reject_, use_ground_truth;
};

// ============================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NDTD2DMapperNode>());
  rclcpp::shutdown();
  return 0;
}