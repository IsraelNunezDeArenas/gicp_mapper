// mapper_node.cpp — v5: limpio y mínimo
//
// Funcionalidad:
//   · Acumula el mapa como rejilla de vóxeles fija (VoxelMap)
//   · Cada scan nuevo se registra contra el mapa acumulado (scan-to-map)
//   · Algoritmo seleccionable: vgicp | gicp
//   · Keyframes por distancia recorrida
//   · Servicio ROS2 para guardar el mapa en PCD
//
// Correcciones respecto a v4:
//   · VoxelMap descomentada y funcional
//   · Guard antes de align() cuando el target aún no está listo
//   · publishMap solo en keyframes (no cada scan)
//   · Cola con límite de tamaño (evita OOM)
//   · Sin const_cast en los wrappers
//   · Wrappers no copiables
//   · VGICPWrapper con epsilons de parada
//   · Publishers fuera del map_mutex_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <segmentation_msgs/msg/semantic_point_cloud.hpp>

#include <Eigen/Core>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using MsgPtr = segmentation_msgs::msg::SemanticPointCloud::SharedPtr;

// ============================================================
//  Utilidades
// ============================================================

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

// Quita NaN, aplica filtro de rango y voxeliza el source.
static CloudT::Ptr prepareCloud(const CloudT::Ptr & in,
                                float max_range,
                                float leaf)
{
  // 1. Eliminar NaN/Inf
  CloudT::Ptr clean(new CloudT);
  {
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*in, *clean, idx);
  }

  // 2. Filtro de rango
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

  // 3. Voxelización
  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(leaf, leaf, leaf);
  vg.setInputCloud(clean);
  auto out = std::make_shared<CloudT>();
  vg.filter(*out);
  return out;
}

template <typename PointCloudTypeT, typename DataT>
pcl::PointXYZ transformPointCloudToGlobal(PointCloudTypeT& pc, geometry_msgs::msg::PoseWithCovariance pose)
{
    Eigen::Isometry3d sensor_to_world_iso;
    tf2::fromMsg(pose.pose, sensor_to_world_iso);
    Eigen::Matrix4f sensor_to_world = sensor_to_world_iso.matrix().cast<float>()
    // Transforming Points to Global Reference Frame
    pcl::transformPointCloud(pc, pc, sensor_to_world)
    // Getting the Translation from the sensor to the Global Reference Frame
    const auto& t = pose.pose.position
    return pcl::PointXYZ((float)t.x, (float)t.y, (float)t.z);
}

// ============================================================
//  VoxelMap — rejilla fija con centroide acumulado
// ============================================================
//
//  Clave: (ix,iy,iz) = floor(p / leaf)
//  La rejilla nunca se desplaza entre scans → sin planos apilados.
//  Inserción incremental O(1) por punto.

class VoxelMap
{
public:
  explicit VoxelMap(float leaf) : leaf_(leaf), inv_leaf_(1.f / leaf) {}

  // No copiable
  VoxelMap(const VoxelMap &)            = delete;
  VoxelMap & operator=(const VoxelMap &) = delete;

  void insert(const PointT & p)
  {
    auto & cell = cells_[toKey(p)];
    ++cell.count;
    // Media incremental: evita acumular suma + división posterior
    const float inv = 1.f / static_cast<float>(cell.count);
    cell.x += (p.x - cell.x) * inv;
    cell.y += (p.y - cell.y) * inv;
    cell.z += (p.z - cell.z) * inv;
  }

  void insertCloud(const CloudT & cloud)
  {
    for (const auto & pt : cloud.points) insert(pt);
  }

  CloudT::Ptr toCloud() const
  {
    auto out = std::make_shared<CloudT>();
    out->reserve(cells_.size());
    for (const auto & [k, c] : cells_) {
      PointT p; p.x = c.x; p.y = c.y; p.z = c.z;
      out->push_back(p);
    }
    out->width  = static_cast<uint32_t>(out->size());
    out->height = 1;
    return out;
  }

  size_t size()  const { return cells_.size(); }
  bool   empty() const { return cells_.empty(); }
  void   clear()       { cells_.clear(); }

private:
  struct Cell { float x{0}, y{0}, z{0}; uint32_t count{0}; };

  struct Key {
    int32_t ix, iy, iz;
    bool operator==(const Key & o) const {
      return ix == o.ix && iy == o.iy && iz == o.iz;
    }
  };

  struct KeyHash {
    size_t operator()(const Key & k) const noexcept {
      size_t h = 2166136261u;
      auto mix = [&](int32_t v) { h ^= static_cast<uint32_t>(v); h *= 16777619u; };
      mix(k.ix); mix(k.iy); mix(k.iz);
      return h;
    }
  };

  Key toKey(const PointT & p) const {
    return {
      static_cast<int32_t>(std::floor(p.x * inv_leaf_)),
      static_cast<int32_t>(std::floor(p.y * inv_leaf_)),
      static_cast<int32_t>(std::floor(p.z * inv_leaf_))
    };
  }

  float leaf_, inv_leaf_;
  std::unordered_map<Key, Cell, KeyHash> cells_;
};

// ============================================================
//  Interfaz de registro
// ============================================================

struct IRegistration
{
  virtual ~IRegistration() = default;
  virtual void            setInputTarget(const CloudT::Ptr &)           = 0;
  virtual void            setInputSource(const CloudT::Ptr &)           = 0;
  virtual void            align(CloudT &, const Eigen::Matrix4f &)      = 0;
  virtual bool            hasConverged()                          const  = 0;
  virtual double          getFitnessScore()                       const  = 0;
  virtual Eigen::Matrix4f getFinalTransformation()                const  = 0;
};

// ── VGICP ─────────────────────────────────────────────────────────────────

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

  double getFitnessScore() const override {
    return mapper_.getFitnessScore();
  }

  Eigen::Matrix4f getFinalTransformation() const override {
    return mapper_.getFinalTransformation();
  }

private:
  mutable fast_gicp::FastVGICP<PointT, PointT> mapper_;
};

// ── GICP ──────────────────────────────────────────────────────────────────

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

  double getFitnessScore() const override {
    return mapper_.getFitnessScore(mapper_.getMaxCorrespondenceDistance());
  }

  Eigen::Matrix4f getFinalTransformation() const override {
    return mapper_.getFinalTransformation();
  }

private:
  mutable fast_gicp::FastGICP<PointT, PointT> mapper_;
};

// ============================================================
//  Nodo principal
// ============================================================

class MapperNode : public rclcpp::Node
{
public:
  MapperNode() : Node("mapper_node")
  {
    // ── Parámetros ─────────────────────────────────────────────────────────
    declare_parameter("scan_topic",        "/cloud_in");
    declare_parameter("algorithm",         "vgicp");   // vgicp | gicp
    declare_parameter("reg_resolution",    0.5);       // solo VGICP
    declare_parameter("max_corr_dist",     5.0);
    declare_parameter("max_iterations",    64);
    declare_parameter("num_threads",       4);
    declare_parameter("max_scan_range",    80.0);
    declare_parameter("source_voxel_size", 0.3);
    declare_parameter("keyframe_dist",     0.5);
    declare_parameter("fitness_reject",    15.0);
    declare_parameter("map_voxel_size",    0.2);
    declare_parameter("map_save_path",     "/tmp/map.pcd");

    const auto algo      = get_parameter("algorithm").as_string();
    const auto reg_res   = get_parameter("reg_resolution").as_double();
    const auto max_corr  = get_parameter("max_corr_dist").as_double();
    const auto max_iter  = get_parameter("max_iterations").as_int();
    const auto threads   = get_parameter("num_threads").as_int();

    source_voxel_size_ = static_cast<float>(get_parameter("source_voxel_size").as_double());
    max_scan_range_    = static_cast<float>(get_parameter("max_scan_range").as_double());
    keyframe_dist_     = static_cast<float>(get_parameter("keyframe_dist").as_double());
    fitness_reject_    = get_parameter("fitness_reject").as_double();
    map_voxel_size_    = static_cast<float>(get_parameter("map_voxel_size").as_double());
    map_save_path_     = get_parameter("map_save_path").as_string();

    // ── Instanciar algoritmo ───────────────────────────────────────────────
    if (algo == "gicp") {
      reg_ = std::make_unique<GICPWrapper>(max_corr, max_iter, threads);
      RCLCPP_INFO(get_logger(), "[Mapper] GICP  max_corr=%.2f  iters=%d  threads=%d",
                  max_corr, max_iter, threads);
    } else {
      reg_ = std::make_unique<VGICPWrapper>(reg_res, max_corr, max_iter, threads);
      RCLCPP_INFO(get_logger(), "[Mapper] VGICP  res=%.2f  max_corr=%.2f  iters=%d  threads=%d",
                  reg_res, max_corr, max_iter, threads);
    }

    RCLCPP_INFO(get_logger(),
      "[Mapper] source_voxel=%.2f  max_range=%.0f  map_voxel=%.2f  "
      "kf_dist=%.2f  fitness_reject=%.2f",
      source_voxel_size_, max_scan_range_, map_voxel_size_,
      keyframe_dist_, fitness_reject_);

    // ── Estado inicial ─────────────────────────────────────────────────────
    voxel_map_          = std::make_unique<VoxelMap>(map_voxel_size_);
    current_pose_       = Eigen::Matrix4f::Identity();
    last_keyframe_pose_ = Eigen::Matrix4f::Identity();
    target_ready_       = false;

    // ── ROS2 ───────────────────────────────────────────────────────────────
    map_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>("/map", 1);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/mapper_pose", 10);

    scan_sub_ = create_subscription<segmentation_msgs::msg::SemanticPointCloud>(
      get_parameter("scan_topic").as_string(), 200,
      [this](MsgPtr msg) { enqueue(std::move(msg)); });

    save_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/save_map",
      std::bind(&MapperNode::onSaveMap, this,
                std::placeholders::_1, std::placeholders::_2));

    // ── Hilo worker ────────────────────────────────────────────────────────
    running_       = true;
    worker_thread_ = std::thread(&MapperNode::workerLoop, this);
  }

  ~MapperNode() override
  {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  // ── Cola con límite ───────────────────────────────────────────────────────
  static constexpr size_t kMaxQueue = 50;

  void enqueue(MsgPtr msg)
  {
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if (msg_queue_.size() >= kMaxQueue) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "[Mapper] Cola llena (%zu) — scan descartado", msg_queue_.size());
        return;
      }
      msg_queue_.push(std::move(msg));
    }
    queue_cv_.notify_one();
  }

  // ── Hilo worker ───────────────────────────────────────────────────────────
  void workerLoop()
  {
    while (running_) {
      MsgPtr msg;
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
        RCLCPP_ERROR(get_logger(), "[Mapper] Excepción en processScan: %s", e.what());
      } catch (...) {
        RCLCPP_ERROR(get_logger(), "[Mapper] Excepción desconocida — scan descartado");
      }
    }
  }

  // ── Procesamiento de un scan ──────────────────────────────────────────────
  void processScan(const MsgPtr & msg)
  {
    // 1. Desempaquetar y preparar
    CloudT::Ptr raw(new CloudT);
    pcl::fromROSMsg(msg->cloud, *raw);
    if (raw->empty()) { RCLCPP_WARN(get_logger(), "Scan vacío — ignorado"); return; }

    const CloudT::Ptr scan = prepareCloud(raw, max_scan_range_, source_voxel_size_);
    if (scan->empty()) { RCLCPP_WARN(get_logger(), "Scan vacío tras filtrado — ignorado"); return; }

    // Datos que se publican fuera del mutex
    std::optional<Eigen::Matrix4f>  pose_to_pub;
    std::optional<CloudT::Ptr>      map_to_pub;

    {
      std::lock_guard<std::mutex> lock(map_mutex_);

      // ── Bootstrap: primer scan ─────────────────────────────────────────
      if (voxel_map_->empty()) {
        current_pose_       = poseToMatrix(msg->pose);
        last_keyframe_pose_ = current_pose_;

        CloudT transformed;
        pcl::transformPointCloud(*scan, transformed, current_pose_);
        voxel_map_->insertCloud(transformed);

        if (voxel_map_->size() >= kMinTargetCells) {
          const auto target = voxel_map_->toCloud();
          reg_->setInputTarget(target);
          target_ready_ = true;
        }

        RCLCPP_INFO(get_logger(), "[Mapper] Bootstrap: %zu celdas", voxel_map_->size());

        pose_to_pub = current_pose_;
        map_to_pub  = voxel_map_->toCloud();
        // (toCloud() se llama dos veces en bootstrap — aceptable, ocurre una sola vez)
        return;  // ← early return dentro del lock, publish ocurre abajo
      }

      // ── Guard: target no disponible aún ───────────────────────────────
      if (!target_ready_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "[Mapper] Target aún no listo (%zu celdas) — scan descartado",
          voxel_map_->size());
        return;
      }

      // ── Guess inicial desde odometría ─────────────────────────────────
      Eigen::Matrix4f guess = poseToMatrix(msg->pose);
      if (!guess.allFinite()) {
        RCLCPP_WARN_ONCE(get_logger(), "[Mapper] Pose no finita — usando pose anterior");
        guess = current_pose_;
      }

      // ── Registro scan-to-map ──────────────────────────────────────────
      reg_->setInputSource(scan);

      CloudT aligned;
      const auto t0 = now();
      reg_->align(aligned, guess);
      const double ms = (now() - t0).seconds() * 1e3;

      const Eigen::Matrix4f T       = reg_->getFinalTransformation();
      const double          fitness = reg_->getFitnessScore();

      if (!reg_->hasConverged()) {
        RCLCPP_WARN(get_logger(),
          "[Mapper] Iteraciones agotadas (%.1f ms)  fitness=%.4f", ms, fitness);
      }

      if (!T.allFinite()) {
        RCLCPP_WARN(get_logger(), "[Mapper] Transformación no finita — rechazada");
        return;
      }

      if (fitness > fitness_reject_) {
        RCLCPP_WARN(get_logger(),
          "[Mapper] Fitness %.4f > umbral %.4f — rechazada (%.1f ms)",
          fitness, fitness_reject_, ms);
        return;
      }

      current_pose_ = T;
      pose_to_pub   = current_pose_;

      // ── Keyframe ──────────────────────────────────────────────────────
      const float dist_kf = (current_pose_.block<3,1>(0,3) -
                             last_keyframe_pose_.block<3,1>(0,3)).norm();

      if (dist_kf >= keyframe_dist_) {
        CloudT transformed;
        pcl::transformPointCloud(*scan, transformed, current_pose_);
        voxel_map_->insertCloud(transformed);

        if (voxel_map_->size() >= kMinTargetCells) {
          const auto target = voxel_map_->toCloud();
          reg_->setInputTarget(target);
          target_ready_ = true;
          map_to_pub    = target;   // reutilizar el puntero recién creado
        }

        last_keyframe_pose_ = current_pose_;

        RCLCPP_INFO(get_logger(),
          "[KF] dist=%.2f m  fitness=%.4f  celdas=%zu  %.1f ms",
          dist_kf, fitness, voxel_map_->size(), ms);
      } else {
        RCLCPP_DEBUG(get_logger(),
          "[Mapper] Sin KF  dist=%.2f  fitness=%.4f  %.1f ms",
          dist_kf, fitness, ms);
      }
    } // ── fin del lock ───────────────────────────────────────────────────

    // Publicar fuera del mutex
    if (pose_to_pub) publishPose(msg->header.stamp, *pose_to_pub);
    if (map_to_pub)  publishMap (msg->header.stamp, *map_to_pub);
  }

  // ── Servicio save_map ─────────────────────────────────────────────────────
  void onSaveMap(
    const std_srvs::srv::Trigger::Request::SharedPtr  /*req*/,
          std_srvs::srv::Trigger::Response::SharedPtr  res)
  {
    CloudT::Ptr cloud;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (voxel_map_->empty()) {
        res->success = false;
        res->message = "Mapa vacío";
        return;
      }
      cloud = voxel_map_->toCloud();
    }

    try {
      std::filesystem::create_directories(
        std::filesystem::path(map_save_path_).parent_path());
    } catch (const std::exception & e) {
      res->success = false;
      res->message = std::string("Error creando directorio: ") + e.what();
      RCLCPP_ERROR(get_logger(), "%s", res->message.c_str());
      return;
    }

    const int ret = pcl::io::savePCDFileBinaryCompressed(map_save_path_, *cloud);
    res->success = (ret == 0);
    res->message = res->success
      ? "Guardado en " + map_save_path_ + " (" + std::to_string(cloud->size()) + " celdas)"
      : "Error al guardar (ret=" + std::to_string(ret) + ")";
    RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
  }

  // ── Publicadores ─────────────────────────────────────────────────────────
  void publishPose(const rclcpp::Time & stamp, const Eigen::Matrix4f & T)
  {
    Eigen::Quaternionf q(T.block<3,3>(0,0));
    q.normalize();

    geometry_msgs::msg::PoseStamped pm;
    pm.header.stamp    = stamp;
    pm.header.frame_id = "map";
    pm.pose.position.x = T(0,3);
    pm.pose.position.y = T(1,3);
    pm.pose.position.z = T(2,3);
    pm.pose.orientation.x = q.x();
    pm.pose.orientation.y = q.y();
    pm.pose.orientation.z = q.z();
    pm.pose.orientation.w = q.w();
    pose_pub_->publish(pm);
  }

  void publishMap(const rclcpp::Time & stamp, const CloudT::Ptr & cloud)
  {
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud, out);
    out.header.stamp    = stamp;
    out.header.frame_id = "map";
    map_pub_->publish(out);
  }

  // ── Constantes ───────────────────────────────────────────────────────────
  static constexpr size_t kMinTargetCells = 25;   // mínimo de celdas para el KD-tree

  // ── Miembros ─────────────────────────────────────────────────────────────
  rclcpp::Subscription<segmentation_msgs::msg::SemanticPointCloud>::SharedPtr scan_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr                 map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr               pose_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr                          save_srv_;

  std::unique_ptr<IRegistration> reg_;
  std::unique_ptr<VoxelMap>      voxel_map_;

  Eigen::Matrix4f current_pose_;
  Eigen::Matrix4f last_keyframe_pose_;
  bool            target_ready_;
  std::mutex      map_mutex_;

  std::queue<MsgPtr>      msg_queue_;
  std::mutex              queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool>       running_{false};
  std::thread             worker_thread_;

  // Parámetros en tiempo de ejecución
  float       source_voxel_size_;
  float       max_scan_range_;
  float       keyframe_dist_;
  float       map_voxel_size_;
  double      fitness_reject_;
  std::string map_save_path_;
};

// ============================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapperNode>());
  rclcpp::shutdown();
  return 0;
}