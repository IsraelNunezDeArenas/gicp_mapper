// ═══════════════════════════════════════════════════════════════════════════════
//  gicp_mapper_node.cpp
//
//  Suscribe a un topic con QoS RELIABLE / KEEP_LAST para ser compatible
//  con publishers que usen esa política.
//
//  Para evitar saturación (GICP ~50-200 ms, publisher depth=10000) se
//  desacopla la recepción del procesamiento mediante un hilo dedicado y
//  una cola interna acotada (parámetro processing_queue_size).
//  Cuando la cola está llena se descarta el mensaje MÁS ANTIGUO (política
//  "latest-wins"), preservando siempre la nube más reciente.
//
//  Parámetros de QoS / cola:
//    subscriber_depth       depth de la suscripción ROS 2          (def. 50)
//    processing_queue_size  máx. mensajes en cola interna C++       (def.  5)
//
//  Parámetro register_policy:
//    "always"         → registra siempre
//    "converged_only" → solo si GICP convergió
//    "best_fitness"   → solo si convergió y mejora el mejor fitness conocido
// ═══════════════════════════════════════════════════════════════════════════════

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Geometry>

#include <segmentation_msgs/msg/semantic_point_cloud.hpp>

#include "gicp_mapper2.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <chrono>
#include <vector>
#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>

using namespace gicp_mapping;
using MsgT = segmentation_msgs::msg::SemanticPointCloud;

// ── Modos de registro ─────────────────────────────────────────────────────────
enum class RegisterPolicy { Always, ConvergedOnly, BestFitness };

static RegisterPolicy parsePolicy(const std::string & s)
{
  if (s == "converged_only") return RegisterPolicy::ConvergedOnly;
  if (s == "best_fitness")   return RegisterPolicy::BestFitness;
  if (s != "always")
    RCLCPP_WARN(rclcpp::get_logger("gicp_mapper"),
      "register_policy desconocida: '%s'. Usando 'always'.", s.c_str());
  return RegisterPolicy::Always;
}

// ── Conversión PoseWithCovariance → Eigen::Matrix4f ──────────────────────────
static Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::PoseWithCovariance & pwc)
{
  const auto & p = pwc.pose;
  Eigen::Quaternionf q(float(p.orientation.w), float(p.orientation.x),
                       float(p.orientation.y), float(p.orientation.z));
  q.normalize();
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3,3>(0,0) = q.toRotationMatrix();
  m(0,3) = float(p.position.x);
  m(1,3) = float(p.position.y);
  m(2,3) = float(p.position.z);
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────

class GICPMapperNode : public rclcpp::Node
{
public:
  GICPMapperNode()
  : Node("gicp_mapper")
  , best_fitness_(std::numeric_limits<float>::max())
  , registered_count_(0)
  , rejected_count_(0)
  , dropped_count_(0)
  , stop_worker_(false)
  {
    // ── Parámetros ────────────────────────────────────────────────────────────
    declare_parameter("map_frame",  "map");
    declare_parameter("base_frame", "base_link");
    declare_parameter("input_topic", "/cloud_with_pose");

    // QoS del subscriber.
    // RELIABLE + KEEP_LAST para ser compatible con el publisher.
    // subscriber_depth no necesita ser tan grande como el publisher (10000)
    // porque la cola interna C++ (processing_queue_size) absorbe los bursts.
    declare_parameter("subscriber_depth",      50);
    declare_parameter("processing_queue_size",  5);

    // GICP
    declare_parameter("num_threads",             4);
    declare_parameter("max_iterations",          64);
    declare_parameter("max_correspondence_dist", 2.0);
    declare_parameter("transformation_epsilon",  1e-3);

    // Puntos para GICP (conteo fijo, no ratio):
    //   gicp_source_points : nube entrante subsampled como source
    //   gicp_target_points : mapa local acumulado subsampled como target
    // El KD-tree del target se recalcula SOLO al añadir un keyframe,
    // no en cada frame → coste amortizado.
    declare_parameter("gicp_source_points",  5000);
    declare_parameter("gicp_target_points", 20000);

    // Compatibilidad con versiones anteriores (ignorados)
    declare_parameter("input_leaf_size", 0.0);
    declare_parameter("map_leaf_size",   0.0);

    // Keyframes
    declare_parameter("keyframe_delta_trans", 0.30);
    declare_parameter("keyframe_delta_angle", 0.15);
    declare_parameter("local_map_window",     20);

    // Guess externo
    declare_parameter("use_external_guess", true);
    declare_parameter("max_cov_trace",      -1.0);

    // Política de registro
    declare_parameter("register_policy",     "always");
    declare_parameter("best_fitness_margin",  0.0);

    // Salida
    declare_parameter("map_topic",   "/gicp_map");
    declare_parameter("pose_topic",  "/gicp_pose");
    declare_parameter("path_topic",  "/gicp_path");
    declare_parameter("publish_tf",   true);
    declare_parameter("map_period",   2.0);

    // Guardado
    declare_parameter("save_path", "/tmp/gicp_map.pcd");

    // ── Leer parámetros ───────────────────────────────────────────────────────
    map_frame_           = get_parameter("map_frame").as_string();
    base_frame_          = get_parameter("base_frame").as_string();
    publish_tf_          = get_parameter("publish_tf").as_bool();
    use_ext_guess_       = get_parameter("use_external_guess").as_bool();
    max_cov_trace_       = get_parameter("max_cov_trace").as_double();
    register_policy_     = parsePolicy(get_parameter("register_policy").as_string());
    best_fitness_margin_ = static_cast<float>(get_parameter("best_fitness_margin").as_double());
    proc_queue_size_     = static_cast<size_t>(get_parameter("processing_queue_size").as_int());

    // ── Mapper ────────────────────────────────────────────────────────────────
    MapperParams p;
    p.num_threads             = get_parameter("num_threads").as_int();
    p.max_iterations          = get_parameter("max_iterations").as_int();
    p.max_correspondence_dist = get_parameter("max_correspondence_dist").as_double();
    p.transformation_epsilon  = get_parameter("transformation_epsilon").as_double();
    p.gicp_source_points      = get_parameter("gicp_source_points").as_int();
    p.gicp_target_points      = get_parameter("gicp_target_points").as_int();
    p.input_leaf_size         = get_parameter("input_leaf_size").as_double();
    p.map_leaf_size           = get_parameter("map_leaf_size").as_double();
    p.keyframe_delta_trans    = get_parameter("keyframe_delta_trans").as_double();
    p.keyframe_delta_angle    = get_parameter("keyframe_delta_angle").as_double();
    p.local_map_window        = get_parameter("local_map_window").as_int();
    mapper_ = std::make_unique<GICPMapper>(p);

    // ── TF ───────────────────────────────────────────────────────────────────
    tf_buf_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_lst_   = std::make_shared<tf2_ros::TransformListener>(*tf_buf_);
    tf_bcast_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // ── Publishers ────────────────────────────────────────────────────────────
    rclcpp::QoS latched(1);
    latched.transient_local();

    map_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>(
                  get_parameter("map_topic").as_string(), latched);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
                  get_parameter("pose_topic").as_string(), 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>(
                  get_parameter("path_topic").as_string(), 1);

    // ── Subscriber ───────────────────────────────────────────────────────────
    // QoS: RELIABLE + KEEP_LAST, idéntico al publisher Python.
    // Un depth moderado es suficiente porque los mensajes que desbordan
    // el processing_queue_size se descartan (latest-wins) antes de procesar.
    rclcpp::QoS sub_qos(
      rclcpp::KeepLast(get_parameter("subscriber_depth").as_int()));
    sub_qos.reliable();

    cloud_sub_ = create_subscription<MsgT>(
      get_parameter("input_topic").as_string(),
      sub_qos,
      [this](MsgT::SharedPtr msg) { enqueue(std::move(msg)); });

    // ── Servicio save_map ─────────────────────────────────────────────────────
    save_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/save_map",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res) {
        doSave(res);
      });

    // ── Timer mapa ────────────────────────────────────────────────────────────
    map_timer_ = create_wall_timer(
      std::chrono::duration<double>(get_parameter("map_period").as_double()),
      [this] { publishMap(); });

    // ── Hilo de procesamiento ─────────────────────────────────────────────────
    worker_ = std::thread(&GICPMapperNode::workerLoop, this);

    const char * policy_str[] = {"always", "converged_only", "best_fitness"};
    RCLCPP_INFO(get_logger(),
      "[gicp_mapper] OK\n"
      "  topic           : %s\n"
      "  QoS             : RELIABLE / KEEP_LAST(%d)\n"
      "  proc_queue_size : %zu  (latest-wins al desbordar)\n"
      "  register_policy : %s\n"
      "  kf Δt=%.2fm Δa=%.2frad  window=%d  threads=%d",
      get_parameter("input_topic").as_string().c_str(),
      get_parameter("subscriber_depth").as_int(),
      proc_queue_size_,
      policy_str[static_cast<int>(register_policy_)],
      p.keyframe_delta_trans, p.keyframe_delta_angle,
      p.local_map_window, p.num_threads);
  }

  ~GICPMapperNode()
  {
    // Señalizar al hilo que debe terminar y despertar si está esperando
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      stop_worker_ = true;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

private:
  // ── Cola interna: recibe en el hilo ROS, procesa en el hilo worker ────────

  // Llamado desde el callback del subscriber (hilo ROS).
  // Si la cola está llena, descarta el mensaje MÁS ANTIGUO (latest-wins):
  // GICP siempre trabajará con la nube más reciente disponible.
  void enqueue(MsgT::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (msg_queue_.size() >= proc_queue_size_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[queue] Cola llena (%zu/%zu): descartando msg más antiguo. "
        "Total descartados: %u. GICP puede estar sobrecargado.",
        msg_queue_.size(), proc_queue_size_, ++dropped_count_);
      msg_queue_.pop_front();   // descarta el más antiguo
    }
    msg_queue_.push_back(std::move(msg));
    queue_cv_.notify_one();
  }

  // Bucle del hilo de procesamiento: extrae mensajes y llama a handleMsg().
  void workerLoop()
  {
    while (true) {
      MsgT::SharedPtr msg;
      {
        std::unique_lock<std::mutex> lk(queue_mutex_);
        queue_cv_.wait(lk, [this] {
          return stop_worker_ || !msg_queue_.empty();
        });
        if (stop_worker_ && msg_queue_.empty()) break;
        msg = std::move(msg_queue_.front());
        msg_queue_.pop_front();
      }
      handleMsg(std::move(msg));
    }
  }

  // ── Procesamiento del mensaje (ejecuta en el hilo worker) ─────────────────
  void handleMsg(MsgT::SharedPtr msg)
  {
    auto cloud = std::make_shared<CloudT>();
    pcl::fromROSMsg(msg->cloud, *cloud);

    cloud = toBaseFrame(cloud, msg->cloud.header.frame_id,
                        rclcpp::Time(msg->header.stamp));
    if (!cloud) return;

    auto guess = extractGuess(msg->pose, rclcpp::Time(msg->header.stamp));
    process(cloud, rclcpp::Time(msg->header.stamp), guess);
  }

  // ── Guess externo ─────────────────────────────────────────────────────────
  std::optional<Eigen::Matrix4f> extractGuess(
    const geometry_msgs::msg::PoseWithCovariance & pwc,
    const rclcpp::Time &)
  {
    if (!use_ext_guess_) return std::nullopt;

    if (max_cov_trace_ > 0.0) {
      double cov_trace = pwc.covariance[0] + pwc.covariance[7] + pwc.covariance[14];
      if (cov_trace > max_cov_trace_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Guess descartado: traza_cov=%.4f > max=%.4f", cov_trace, max_cov_trace_);
        return std::nullopt;
      }
    }
    return poseToMatrix(pwc);
  }

  // ── Política de registro ──────────────────────────────────────────────────
  bool shouldRegister(const RegistrationResult & result)
  {
    switch (register_policy_) {

      case RegisterPolicy::Always:
        ++registered_count_;
        return true;

      case RegisterPolicy::ConvergedOnly:
        if (!result.converged) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "[policy:converged_only] Descartada — no convergió (fitness=%.4f). "
            "Rechazadas: %u", result.fitness, ++rejected_count_);
          return false;
        }
        ++registered_count_;
        return true;

      case RegisterPolicy::BestFitness:
        if (!result.converged) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "[policy:best_fitness] Descartada — no convergió (fitness=%.4f). "
            "Rechazadas: %u", result.fitness, ++rejected_count_);
          return false;
        }
        if (result.fitness < best_fitness_ - best_fitness_margin_) {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
            "[policy:best_fitness] Nuevo mejor: %.4f → %.4f (Δ=%.4f)",
            best_fitness_, result.fitness, best_fitness_ - result.fitness);
          best_fitness_ = result.fitness;
          ++registered_count_;
          return true;
        }
        RCLCPP_DEBUG(get_logger(),
          "[policy:best_fitness] Descartada — fitness=%.4f no mejora best=%.4f. "
          "Rechazadas: %u", result.fitness, best_fitness_, ++rejected_count_);
        return false;
    }
    return true;
  }

  // ── Mapeo ─────────────────────────────────────────────────────────────────
  void process(const CloudPtr & cloud,
               const rclcpp::Time & stamp,
               const std::optional<Eigen::Matrix4f> & guess)
  {
    if (!cloud || cloud->empty()) return;

    auto result = mapper_->addCloud(cloud, stamp, guess);

    if (!shouldRegister(result)) return;

    if (result.is_keyframe) {
      RCLCPP_INFO(get_logger(),
        "Keyframe #%d  fitness=%.4f  pts=%zu  guess=%s  Δpos=%.3fm  "
        "[reg:%u  rej:%u  drop:%u]",
        mapper_->keyframeCount(), result.fitness, cloud->size(),
        guess.has_value() ? "SI" : "NO",
        guess.has_value()
          ? (result.pose.block<3,1>(0,3) - guess->block<3,1>(0,3)).norm()
          : 0.0f,
        registered_count_, rejected_count_, dropped_count_.load());

      publishPoseAndPath(result.pose, stamp);
    }

    if (publish_tf_) publishTF(result.pose, stamp);
  }

  // ── TF de nube al base_frame ──────────────────────────────────────────────
  CloudPtr toBaseFrame(const CloudPtr & cloud,
                       const std::string & from,
                       const rclcpp::Time & stamp)
  {
    if (from.empty() || from == base_frame_) return cloud;
    try {
      auto tf_stamped = tf_buf_->lookupTransform(
        base_frame_, from, tf2::TimePointZero, tf2::durationFromSec(0.1));
      sensor_msgs::msg::PointCloud2 in_msg, out_msg;
      pcl::toROSMsg(*cloud, in_msg);
      in_msg.header.frame_id = from;
      in_msg.header.stamp    = stamp;
      tf2::doTransform(in_msg, out_msg, tf_stamped);
      auto out = std::make_shared<CloudT>();
      pcl::fromROSMsg(out_msg, *out);
      return out;
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "TF %s→%s: %s", from.c_str(), base_frame_.c_str(), e.what());
      return cloud;
    }
  }

  // ── Publicación ───────────────────────────────────────────────────────────
  void publishMap()
  {
    if (mapper_->keyframeCount() == 0) return;
    auto map = mapper_->buildGlobalMap();
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*map, msg);
    msg.header.stamp    = now();
    msg.header.frame_id = map_frame_;
    map_pub_->publish(msg);
    RCLCPP_DEBUG(get_logger(),
      "Mapa: %zu pts  %d kf  [reg:%u rej:%u drop:%u]",
      map->size(), mapper_->keyframeCount(),
      registered_count_, rejected_count_, dropped_count_.load());
  }

  void publishPoseAndPath(const Eigen::Matrix4f & tf, const rclcpp::Time & stamp)
  {
    Eigen::Affine3d aff(tf.cast<double>());
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp    = stamp;
    ps.header.frame_id = map_frame_;
    ps.pose            = tf2::toMsg(aff);
    pose_pub_->publish(ps);
    trajectory_.push_back(ps);
    nav_msgs::msg::Path path;
    path.header = ps.header;
    path.poses  = trajectory_;
    path_pub_->publish(path);
  }

  void publishTF(const Eigen::Matrix4f & tf, const rclcpp::Time & stamp)
  {
    Eigen::Affine3d aff(tf.cast<double>());
    auto t            = tf2::eigenToTransform(aff);
    t.header.stamp    = stamp;
    t.header.frame_id = map_frame_;
    t.child_frame_id  = base_frame_;
    tf_bcast_->sendTransform(t);
  }

  // ── Guardar mapa ──────────────────────────────────────────────────────────
  void doSave(std_srvs::srv::Trigger::Response::SharedPtr res)
  {
    if (mapper_->keyframeCount() == 0) {
      res->success = false; res->message = "Mapa vacío."; return;
    }

    // buildGlobalMap() devuelve la concatenación de todos los keyframes
    // transformados al frame del mapa. Cada punto conserva su color original.
    // Sin vóxeles: el PCD guardado es la nube completa tal como fue captada.
    auto map  = mapper_->buildGlobalMap();
    auto path = get_parameter("save_path").as_string();
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    if (pcl::io::savePCDFileBinaryCompressed(path, *map) == 0) {
      res->success = true;
      res->message =
        "Guardado: " + path +
        "\n  puntos  : " + std::to_string(map->size()) +
        " (nube completa, color original)" +
        "\n  kf      : " + std::to_string(mapper_->keyframeCount()) +
        "\n  [reg:"    + std::to_string(registered_count_) +
        "  rej:"       + std::to_string(rejected_count_) +
        "  drop:"      + std::to_string(dropped_count_.load()) + "]";
    } else {
      res->success = false;
      res->message = "Error al escribir en: " + path;
    }
    RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
  }

  // ── Miembros ──────────────────────────────────────────────────────────────

  std::unique_ptr<GICPMapper> mapper_;

  std::shared_ptr<tf2_ros::Buffer>               tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener>    tf_lst_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_bcast_;

  rclcpp::Subscription<MsgT>::SharedPtr                         cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr             path_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr            save_srv_;
  rclcpp::TimerBase::SharedPtr                                  map_timer_;

  std::vector<geometry_msgs::msg::PoseStamped> trajectory_;

  // Cola interna desacoplada del hilo ROS
  std::deque<MsgT::SharedPtr>  msg_queue_;
  std::mutex                   queue_mutex_;
  std::condition_variable      queue_cv_;
  std::thread                  worker_;
  bool                         stop_worker_;
  size_t                       proc_queue_size_;

  // Parámetros de comportamiento
  std::string    map_frame_;
  std::string    base_frame_;
  bool           publish_tf_         {true};
  bool           use_ext_guess_      {true};
  double         max_cov_trace_      {-1.0};
  RegisterPolicy register_policy_    {RegisterPolicy::Always};
  float          best_fitness_margin_{0.0f};

  // Contadores de diagnóstico
  float                 best_fitness_;
  uint32_t              registered_count_;
  uint32_t              rejected_count_;
  std::atomic<uint32_t> dropped_count_;
};

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GICPMapperNode>());
  rclcpp::shutdown();
  return 0;
}