// registration_node.cpp
//
// Nodo de REGISTRO scan-to-map (sin mapa local propio).
//
// Funcionalidad:
//   · Suscribe la lectura de la cámara RGBD (SemanticPointCloud: nube + pose + info extra)
//   · Suscribe el MAPA (nube de centroides de vóxel) publicado por OTRO nodo (p.ej. el
//     mapper_node que ya tienes, que sigue siendo el dueño del VoxelMap acumulado)
//   · Registra cada scan contra el último mapa recibido (scan-to-map), con el algoritmo
//     que elijas por parámetro: gicp | vgicp | ndt
//   · Publica el MISMO SemanticPointCloud recibido, pero con la pose corregida por el
//     registro, para que el nodo de mapeo lo integre en su VoxelMap.
//
// Este nodo NO mantiene un VoxelMap propio: el "mapa" contra el que se registra es
// siempre el último que llega por tópico. Si aún no ha llegado ningún mapa, el scan
// se reenvía con su pose original (sin corregir) para permitir el bootstrap del mapa
// en el nodo de mapeo.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <segmentation_msgs/msg/semantic_point_cloud.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

using PointT     = pcl::PointXYZ;
using CloudT     = pcl::PointCloud<PointT>;
using CustomMsg  = segmentation_msgs::msg::SemanticPointCloud;
using CustomPtr  = CustomMsg::SharedPtr;

// ============================================================
//  Utilidades
// ============================================================




double computeInitialFitness(
    const CloudT::Ptr& source,
    const CloudT::Ptr& target,
    const Eigen::Matrix4f& T)
{
    CloudT transformed;
    pcl::transformPointCloud(*source, transformed, T);

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(target);

    double score = 0.0;
    int count = 0;

    std::vector<int> idx(1);
    std::vector<float> dist(1);

    for (const auto& p : transformed.points)
    {
        if (kdtree.nearestKSearch(p, 1, idx, dist) > 0)
        {
            score += dist[0];
            count++;
        }
    }

    return count > 0 ? score / count : std::numeric_limits<double>::max();
}


static Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::PoseWithCovariance & pwc)
{
  const auto & p = pwc.pose;
  Eigen::Quaternionf q(p.orientation.w, p.orientation.x,
                       p.orientation.y, p.orientation.z);
  q.normalize();

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3,3>(0,0) = q.toRotationMatrix();
  T(0,3) = static_cast<float>(p.position.x);
  T(1,3) = static_cast<float>(p.position.y);
  T(2,3) = static_cast<float>(p.position.z);
  return T;
}

static void matrixToPose(const Eigen::Matrix4f & T, geometry_msgs::msg::PoseWithCovariance & pwc)
{
  Eigen::Quaternionf q(T.block<3,3>(0,0));
  q.normalize();
  pwc.pose.position.x = T(0,3);
  pwc.pose.position.y = T(1,3);
  pwc.pose.position.z = T(2,3);
  pwc.pose.orientation.x = q.x();
  pwc.pose.orientation.y = q.y();
  pwc.pose.orientation.z = q.z();
  pwc.pose.orientation.w = q.w();
  // La covarianza original se deja tal cual: este nodo no modela la incertidumbre
  // residual del registro. Si lo necesitas, aquí es donde se inflaría/ajustaría.
}

// Quita NaN, aplica filtro de rango y voxeliza el scan de entrada.
static CloudT::Ptr prepareCloud(const CloudT::Ptr & in, float max_range, float leaf)
{
  CloudT::Ptr clean(new CloudT);
  {
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*in, *clean, idx);
  }

  if (max_range > 0.f) {
    const float r2_min = 0.25f;          // 0.5 m mínimo
    const float r2_max = max_range * max_range;
    CloudT::Ptr ranged(new CloudT);
    ranged->reserve(clean->size());
    for (const auto & pt : clean->points) {
      const float r2 = pt.x*pt.x + pt.y*pt.y + pt.z*pt.z;
      if (r2 >= r2_min && r2 <= r2_max)
        ranged->push_back(pt);
    }
    clean = ranged;
  }

  if (clean->empty() || leaf <= 0.f) return clean;

  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(leaf, leaf, leaf);
  vg.setInputCloud(clean);
  auto out = std::make_shared<CloudT>();
  vg.filter(*out);
  return out;
}

// ============================================================
//  Interfaz de registro
// ============================================================

struct IRegistration
{
  virtual ~IRegistration() = default;
  virtual void            setInputTarget(const CloudT::Ptr &)      = 0;
  virtual void            setInputSource(const CloudT::Ptr &)      = 0;
  virtual void            align(CloudT &, const Eigen::Matrix4f &) = 0;
  virtual bool            hasConverged()                    const = 0;
  virtual double          getFitnessScore()                 const = 0;
  virtual Eigen::Matrix4f getFinalTransformation()           const = 0;
};

// ── VGICP ────────────────────────────────────────────────────────────────

struct VGICPWrapper : public IRegistration
{
  VGICPWrapper(double res, double max_corr, int iters, int threads)
  {
    mapper_.setResolution(static_cast<float>(res));
    mapper_.setMaxCorrespondenceDistance(max_corr);
    mapper_.setMaximumIterations(iters);
    mapper_.setNumThreads(threads);
    mapper_.setTransformationEpsilon(1e-4);
    mapper_.setEuclideanFitnessEpsilon(1e-4);
  }

  VGICPWrapper(const VGICPWrapper &)            = delete;
  VGICPWrapper & operator=(const VGICPWrapper &) = delete;

  void setInputTarget(const CloudT::Ptr & c) override { mapper_.setInputTarget(c); }
  void setInputSource(const CloudT::Ptr & c) override { mapper_.setInputSource(c); }
  void align(CloudT & o, const Eigen::Matrix4f & g) override { mapper_.align(o, g); }
  bool hasConverged() const override { return mapper_.hasConverged(); }
  double getFitnessScore() const override { return mapper_.getFitnessScore(); }
  Eigen::Matrix4f getFinalTransformation() const override { return mapper_.getFinalTransformation(); }

private:
  mutable fast_gicp::FastVGICP<PointT, PointT> mapper_;
};

// ── GICP ─────────────────────────────────────────────────────────────────

struct GICPWrapper : public IRegistration
{
  GICPWrapper(double max_corr, int iters, int threads)
  {
    mapper_.setMaxCorrespondenceDistance(max_corr);
    mapper_.setMaximumIterations(iters);
    mapper_.setNumThreads(threads);
    mapper_.setTransformationEpsilon(1e-4);
    mapper_.setEuclideanFitnessEpsilon(1e-4);
  }

  GICPWrapper(const GICPWrapper &)            = delete;
  GICPWrapper & operator=(const GICPWrapper &) = delete;

  void setInputTarget(const CloudT::Ptr & c) override { mapper_.setInputTarget(c); }
  void setInputSource(const CloudT::Ptr & c) override { mapper_.setInputSource(c); }
  void align(CloudT & o, const Eigen::Matrix4f & g) override { mapper_.align(o, g); }
  bool hasConverged() const override { return mapper_.hasConverged(); }
  double getFitnessScore() const override { return mapper_.getFitnessScore(mapper_.getMaxCorrespondenceDistance()); }
  Eigen::Matrix4f getFinalTransformation() const override { return mapper_.getFinalTransformation(); }

private:
  mutable fast_gicp::FastGICP<PointT, PointT> mapper_;
};

// ── NDT ──────────────────────────────────────────────────────────────────

struct NDTWrapper : public IRegistration
{
  NDTWrapper(double resolution, double step_size, double trans_eps, int iters)
  {
    ndt_.setResolution(static_cast<float>(resolution));
    ndt_.setStepSize(step_size);
    ndt_.setTransformationEpsilon(trans_eps);
    ndt_.setMaximumIterations(iters);
  }

  NDTWrapper(const NDTWrapper &)            = delete;
  NDTWrapper & operator=(const NDTWrapper &) = delete;

  void setInputTarget(const CloudT::Ptr & c) override { ndt_.setInputTarget(c); }
  void setInputSource(const CloudT::Ptr & c) override { ndt_.setInputSource(c); }
  void align(CloudT & o, const Eigen::Matrix4f & g) override { ndt_.align(o, g); }
  bool hasConverged() const override { return ndt_.hasConverged(); }
  double getFitnessScore() const override { return ndt_.getFitnessScore(); }
  Eigen::Matrix4f getFinalTransformation() const override { return ndt_.getFinalTransformation(); }

private:
  mutable pcl::NormalDistributionsTransform<PointT, PointT> ndt_;
};

// ============================================================
//  Nodo principal
// ============================================================

class RegistrationNode : public rclcpp::Node
{
public:
  RegistrationNode() : Node("registration_node")
  {
    // ── Parámetros ─────────────────────────────────────────────────────────
    declare_parameter("scan_topic",       "cloud_in");         // Lectura
    declare_parameter("map_topic",        "bonxai_point_cloud_centers");      // Mapa
    declare_parameter("corrected_topic",  "cloud_registered"); // Salida corregida
    declare_parameter("pose_topic",       "registered_pose"); // PoseStamped (solo debug/RViz)

    declare_parameter("algorithm",         "vgicp");   // vgicp | gicp | ndt
    declare_parameter("reg_resolution",    0.5);       // VGICP
    declare_parameter("max_corr_dist",     5.0);        // GICP / VGICP
    declare_parameter("max_iterations",    64);
    declare_parameter("num_threads",       8);          // GICP / VGICP
    declare_parameter("ndt_resolution",    1.0);        // NDT
    declare_parameter("ndt_step_size",     0.1);        // NDT
    declare_parameter("ndt_trans_eps",     0.01);       // NDT

    declare_parameter("max_scan_range",    80.0);
    declare_parameter("source_voxel_size", 0.3);
    declare_parameter("fitness_reject",    15.0);
    declare_parameter("drop_on_reject",    false); // false: reenvía con pose sin corregir

    const auto algo     = get_parameter("algorithm").as_string();
    const auto reg_res  = get_parameter("reg_resolution").as_double();
    const auto max_corr = get_parameter("max_corr_dist").as_double();
    const auto max_iter = static_cast<int>(get_parameter("max_iterations").as_int());
    const auto threads  = static_cast<int>(get_parameter("num_threads").as_int());
    const auto ndt_res  = get_parameter("ndt_resolution").as_double();
    const auto ndt_step = get_parameter("ndt_step_size").as_double();
    const auto ndt_eps  = get_parameter("ndt_trans_eps").as_double();

    max_scan_range_     = static_cast<float>(get_parameter("max_scan_range").as_double());
    source_voxel_size_  = static_cast<float>(get_parameter("source_voxel_size").as_double());
    fitness_reject_     = get_parameter("fitness_reject").as_double();
    drop_on_reject_     = get_parameter("drop_on_reject").as_bool();

    RCLCPP_INFO(
    get_logger(),
    "NDT resolution double=%.10f",
    ndt_res
);

    // ── Instanciar algoritmo ───────────────────────────────────────────────
    if (algo == "gicp") {
      reg_ = std::make_unique<GICPWrapper>(max_corr, max_iter, threads);
      RCLCPP_INFO(get_logger(), "[Reg] GICP  max_corr=%.4f iters=%d threads=%d",
                  max_corr, max_iter, threads);
    } else if (algo == "ndt") {
      reg_ = std::make_unique<NDTWrapper>(ndt_res, ndt_step, ndt_eps, max_iter);
      RCLCPP_INFO(get_logger(), "[Reg] NDT  res=%.4f step=%.2f eps=%.4f iters=%d",
                  ndt_res, ndt_step, ndt_eps, max_iter);
    } else {
      reg_ = std::make_unique<VGICPWrapper>(reg_res, max_corr, max_iter, threads);
      RCLCPP_INFO(get_logger(), "[Reg] VGICP  res=%.4f max_corr=%.2f iters=%d threads=%d",
                  reg_res, max_corr, max_iter, threads);
    }

    // ── ROS2 ───────────────────────────────────────────────────────────────
    cloud_pub_ = create_publisher<CustomMsg>(
      get_parameter("corrected_topic").as_string(), 50);
    pose_pub_  = create_publisher<geometry_msgs::msg::PoseStamped>(
      get_parameter("pose_topic").as_string(), 10);

    scan_sub_ = create_subscription<CustomMsg>(
      get_parameter("scan_topic").as_string(), 200,
      [this](CustomPtr msg) { enqueue(std::move(msg)); });

    map_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      get_parameter("map_topic").as_string(),
      rclcpp::QoS(1),
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { onMap(std::move(msg)); });

    // ── Hilo worker (para no bloquear el callback de suscripción) ──────────
    running_       = true;
    worker_thread_ = std::thread(&RegistrationNode::workerLoop, this);

    RCLCPP_INFO(get_logger(),
      "[Reg] scan_topic=%s  map_topic=%s  corrected_topic=%s  (sin mapa local propio)",
      get_parameter("scan_topic").as_string().c_str(),
      get_parameter("map_topic").as_string().c_str(),
      get_parameter("corrected_topic").as_string().c_str());
  }

  ~RegistrationNode() override
  {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  static constexpr size_t kMaxQueue = 50;

  // ── Recepción del mapa externo (del otro nodo) ──────────────────────────
  void onMap(sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (static_cast<size_t>(msg->width) * msg->height == 0) return;

    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    std::lock_guard<std::mutex> lock(map_mutex_);
    map_cloud_    = cloud;
    map_dirty_    = true;
    target_ready_ = true;
  }

  // ── Cola con límite ──────────────────────────────────────────────────────
  void enqueue(CustomPtr msg)
  {
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if (msg_queue_.size() >= kMaxQueue) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "[Reg] Cola llena (%zu) — scan descartado", msg_queue_.size());
        return;
      }
      msg_queue_.push(std::move(msg));
    }
    queue_cv_.notify_one();
  }

  void workerLoop()
  {
    while (running_) {
      CustomPtr msg;
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
        RCLCPP_ERROR(get_logger(), "[Reg] Excepción en processScan: %s", e.what());
      } catch (...) {
        RCLCPP_ERROR(get_logger(), "[Reg] Excepción desconocida — scan descartado");
      }
    }
  }

  // ── Procesamiento de un scan ─────────────────────────────────────────────
  void processScan(const CustomPtr & msg)
  {
    CloudT::Ptr raw(new CloudT);
    pcl::fromROSMsg(msg->cloud, *raw);
    if (raw->empty()) { RCLCPP_WARN(get_logger(), "[Reg] Scan vacío — ignorado"); return; }

    const CloudT::Ptr scan = prepareCloud(raw, max_scan_range_, source_voxel_size_);
    if (scan->empty()) { RCLCPP_WARN(get_logger(), "[Reg] Scan vacío tras filtrado — ignorado"); return; }

    Eigen::Matrix4f guess = poseToMatrix(msg->pose);
    if (!guess.allFinite()) {
      RCLCPP_WARN(get_logger(), "[Reg] Pose de entrada no finita — scan descartado");
      return;
    }

    Eigen::Matrix4f corrected = guess;   // por defecto: sin corregir (fallback / bootstrap)
    bool did_register = false;

    {
      std::lock_guard<std::mutex> lock(map_mutex_);

      if (!target_ready_ || !map_cloud_ || map_cloud_->empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
          "[Reg] Mapa aún no disponible — reenviando pose sin corregir (bootstrap)");
      } else {
        // Reconstruir el target solo cuando el mapa haya cambiado (evita rehacer el KD-tree cada scan)
        if (map_dirty_) {
          reg_->setInputTarget(map_cloud_);
          map_dirty_ = false;
        }

        reg_->setInputSource(scan);

        const double pre_score = computeInitialFitness(scan,map_cloud_,guess);

        CloudT aligned;
        const auto t0 = now();
        reg_->align(aligned, guess);
        const double ms = (now() - t0).seconds() * 1e3;

        const Eigen::Matrix4f T       = reg_->getFinalTransformation();
        const double          fitness = reg_->getFitnessScore();
        const bool            ok = reg_->hasConverged() && T.allFinite() && fitness <= fitness_reject_;

        if (ok || pre_score > fitness) {
          corrected     = T;
          did_register  = true;
          RCLCPP_INFO(get_logger(), "[Reg] OK  fitness=%.4f  %.2f ms", fitness, ms);
        }
        else {
          RCLCPP_WARN(get_logger(),
            "[Reg] Rechazado (converged=%d fitness=%.4f finite=%d)  %.2f ms",
            reg_->hasConverged(), fitness, T.allFinite(), ms);
          if (drop_on_reject_) return;
          // si no se descarta, se reenvía con 'guess' (pose original) tal cual
        }
      }
    }

    publishCorrected(msg, corrected, did_register);
  }

  // ── Publicación ──────────────────────────────────────────────────────────
  void publishCorrected(const CustomPtr & msg, const Eigen::Matrix4f & T, bool /*did_register*/)
  {
    // Copiamos el mensaje completo (nube + info semántica/adicional intacta)
    // y solo sobrescribimos la pose con la corregida.
    auto out = std::make_shared<CustomMsg>(*msg);
    matrixToPose(T, out->pose);
    cloud_pub_->publish(*out);

    if (pose_pub_->get_subscription_count() > 0) {
      geometry_msgs::msg::PoseStamped pm;
      pm.header = msg->header;
      pm.header.frame_id = "map";
      pm.pose = out->pose.pose;
      pose_pub_->publish(pm);
    }
  }

  // ── Miembros ─────────────────────────────────────────────────────────────
  rclcpp::Subscription<CustomMsg>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Publisher<CustomMsg>::SharedPtr cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  std::unique_ptr<IRegistration> reg_;

  // Mapa externo (no acumulado aquí, solo el último recibido)
  CloudT::Ptr map_cloud_;
  bool        map_dirty_    = false;
  bool        target_ready_ = false;
  std::mutex  map_mutex_;

  std::queue<CustomPtr>    msg_queue_;
  std::mutex               queue_mutex_;
  std::condition_variable  queue_cv_;
  std::atomic<bool>        running_{false};
  std::thread              worker_thread_;

  // Parámetros en tiempo de ejecución
  float  max_scan_range_;
  float  source_voxel_size_;
  double fitness_reject_;
  bool   drop_on_reject_;
};

// ============================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RegistrationNode>());
  rclcpp::shutdown();
  return 0;
}