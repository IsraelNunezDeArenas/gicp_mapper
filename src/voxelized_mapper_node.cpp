#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <segmentation_msgs/msg/semantic_point_cloud.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <std_srvs/srv/trigger.hpp>

#include <Eigen/Core>
#include <deque>
#include <mutex>

using PointT    = pcl::PointXYZ;
using CloudT    = pcl::PointCloud<PointT>;
using CloudTPtr = CloudT::Ptr;
using CustomMsg = segmentation_msgs::msg::SemanticPointCloud;

static Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose & p)
{
  const Eigen::Quaternionf q(p.orientation.w, p.orientation.x,
                              p.orientation.y, p.orientation.z);
  Eigen::Matrix4f T   = Eigen::Matrix4f::Identity();
  T.block<3,3>(0,0)   = q.normalized().toRotationMatrix();
  T.block<3,1>(0,3)   = Eigen::Vector3f(p.position.x, p.position.y, p.position.z);
  return T;
}

class GicpMappingNode : public rclcpp::Node
{
public:
  GicpMappingNode() : Node("gicp_mapping_node")
  {
    declare_parameter("use_vgicp",              true);
    declare_parameter("leaf_size_input",         0.2);
    declare_parameter("leaf_size_map",           0.2);
    declare_parameter("max_range",               50.0);
    declare_parameter("min_range",               0.5);
    declare_parameter("max_iterations",          64);
    declare_parameter("transformation_epsilon",  1e-3);
    declare_parameter("rotation_epsilon",  1e-3);
    declare_parameter("max_correspondence_dist", 2.0);
    declare_parameter("num_threads",             4);
    declare_parameter("vgicp_resolution",        1.0);
    declare_parameter("map_max_frames",          200);
    declare_parameter("keyframe_dist",           0.5);
    declare_parameter("keyframe_angle",          0.087);
    declare_parameter("force_accept",            true);
    declare_parameter("score_warn_threshold",    1.0);
    declare_parameter("save_voxel_size",         0.02);
    declare_parameter("map_save_path",  std::string("/home/ubuntu/map"));
    declare_parameter("map_save_format",std::string("pcd"));
    declare_parameter("local_window_size", 5);

    use_vgicp_            = get_parameter("use_vgicp").as_bool();
    leaf_size_input_      = get_parameter("leaf_size_input").as_double();
    leaf_size_map_        = get_parameter("leaf_size_map").as_double();
    max_range_            = get_parameter("max_range").as_double();
    min_range_            = get_parameter("min_range").as_double();
    max_iterations_       = get_parameter("max_iterations").as_int();
    transf_epsilon_       = get_parameter("transformation_epsilon").as_double();
    rot_epsilon_          = get_parameter("rotation_epsilon").as_double();
    max_corr_dist_        = get_parameter("max_correspondence_dist").as_double();
    num_threads_          = get_parameter("num_threads").as_int();
    vgicp_resolution_     = get_parameter("vgicp_resolution").as_double();
    map_max_frames_       = (std::size_t)get_parameter("map_max_frames").as_int();
    keyframe_dist_        = get_parameter("keyframe_dist").as_double();
    keyframe_angle_       = get_parameter("keyframe_angle").as_double();
    force_accept_         = get_parameter("force_accept").as_bool();
    score_warn_threshold_ = get_parameter("score_warn_threshold").as_double();
    save_voxel_size_      = get_parameter("save_voxel_size").as_double();
    save_map_path_        = get_parameter("map_save_path").as_string();
    file_map_             = get_parameter("map_save_format").as_string();

    local_window_size_       = (std::size_t)get_parameter("local_window_size").as_int();


    // FIX 1: solo shared_ptr, sin miembro de valor duplicado
    if (use_vgicp_) {
      vgicp_ = std::make_shared<fast_gicp::FastVGICP<PointT, PointT>>();
      vgicp_->setMaximumIterations(max_iterations_);
      vgicp_->setTransformationEpsilon(transf_epsilon_);
      vgicp_->setTransformationRotationEpsilon(rot_epsilon_);
      vgicp_->setMaxCorrespondenceDistance(max_corr_dist_);
      vgicp_->setResolution(vgicp_resolution_);
      vgicp_->setNumThreads(num_threads_);
    } else {
      gicp_ = std::make_shared<fast_gicp::FastGICP<PointT, PointT>>();
      gicp_->setMaximumIterations(max_iterations_);
      gicp_->setTransformationEpsilon(transf_epsilon_);
      gicp_->setTransformationRotationEpsilon(rot_epsilon_);
      gicp_->setMaxCorrespondenceDistance(max_corr_dist_);
      gicp_->setNumThreads(num_threads_);
    }

    voxel_in_.setLeafSize(leaf_size_input_, leaf_size_input_, leaf_size_input_);
    voxel_map_.setLeafSize(leaf_size_map_ , leaf_size_map_,  leaf_size_map_);

    voxel_local_window_.setLeafSize(leaf_size_input_,
                          leaf_size_input_,
                          leaf_size_input_);

    map_cloud_ = std::make_shared<CloudT>();

    retry_count = 0;

    const auto qos = rclcpp::QoS(5).reliable();
    pub_map_     = create_publisher<sensor_msgs::msg::PointCloud2>("/gicp/map",     qos);
    pub_aligned_ = create_publisher<sensor_msgs::msg::PointCloud2>("/gicp/aligned", qos);

    sub_ = create_subscription<CustomMsg>(
      "/cloud_in", rclcpp::SensorDataQoS(),
      [this](const CustomMsg::SharedPtr msg) { onCloud(msg); });

    // FIX 3: srv_save_map_ dentro del constructor
    srv_save_map_ = create_service<std_srvs::srv::Trigger>(
      "/gicp/save_map",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res)
      {
        // std::lock_guard<std::mutex> lk(mtx_);
        if (map_cloud_->empty()) {
          res->success = false;
          res->message = "Mapa vacío, nada que guardar";
          return;
        }

        auto map_voxelized = std::make_shared<CloudT>();
        voxel_map_.setInputCloud(map_cloud_);
        voxel_map_.filter(*map_voxelized);

        int ret = -1;
        std::string full_path;
        if (file_map_ == "ply") {
          full_path = save_map_path_ + ".ply";
          ret = pcl::io::savePLYFileBinary(full_path, *map_voxelized);
        } else {
          full_path = save_map_path_ + ".pcd";
          ret = pcl::io::savePCDFileBinary(full_path, *map_voxelized);
        }

        res->success = (ret == 0);
        res->message = res->success
          ? "Mapa guardado en " + full_path + " (" + std::to_string(map_voxelized->size()) + " pts)"
          : "Error al guardar en " + full_path;
        RCLCPP_INFO(get_logger(), "[GicpMapper] %s", res->message.c_str());
      });

    RCLCPP_INFO(get_logger(), "[GicpMapper] %s listo", use_vgicp_ ? "VGICP" : "GICP");
  }  // ← cierre real del constructor

private:

  void onCloud(const CustomMsg::SharedPtr msg)
{
  Eigen::Matrix4f refined_T; 
  auto input = preprocess(msg->cloud);
  if (input->empty()) return;

  const Eigen::Matrix4f T_map_sensor = poseToEigen(msg->pose.pose);

  // Primer frame: inicializar ventana
  if (!initialized_) {
    auto first = std::make_shared<CloudT>();
    pcl::transformPointCloud(*input, *first, T_map_sensor);

    refined_pose_ = T_map_sensor;
    kf_pose_      = T_map_sensor;
    initialized_  = true;

    local_window_.push_back(first);   // primer elemento de la ventana
    addToMap(first);

    RCLCPP_INFO(get_logger(), "[GicpMapper] Inicializado con %zu pts", first->size());
    publishAll(msg->header, refined_pose_, first);
    return;
  }

  // Construir target desde la ventana local
  CloudTPtr local_target;
  local_target = buildLocalTarget();

  if (local_target->empty()) return;

  // Registro GICP contra la ventana
  auto aligned = std::make_shared<CloudT>();
  
  double score     = 0.0;
  double pre_score = 0.0;
  bool   converged = false;

  {
    auto run_registration = [&](auto & reg) {
      reg.setInputSource(input);
      reg.setInputTarget(local_target);   // ← ventana concatenada
      pre_score = reg.getFitnessScore();

      reg.align(*aligned, T_map_sensor);
      refined_T = reg.getFinalTransformation();
      score     = reg.getFitnessScore();
      converged = reg.hasConverged();
    };

    if (use_vgicp_) run_registration(*vgicp_);
    else            run_registration(*gicp_);
  }

  RCLCPP_INFO(get_logger(),
    "[GicpMapper] fitness=%.4f converged=%d window=%zu",
    score, converged, local_window_.size());

  std::stringstream ss;
  ss << refined_T;

  RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());

  // Validar convergencia
  const bool accepted = (score <= score_warn_threshold_);
  if (!accepted) {              // Score no aceptable, nube muy alejada
    if (!force_accept_) return; // Si no se obliga a aceptar nube, salta

    if (pre_score > score || pre_score >= 5 * score_warn_threshold_){ 

      if (retry_count < 5){
        retry_count++; 
        return; 
      }  

    }
    retry_count = 0;
    refined_T = T_map_sensor;
    RCLCPP_INFO(get_logger(), "[GicpMapper] Prescore  %.3f -> Usando pose inicial",pre_score);
    
    pcl::transformPointCloud(*input, *aligned, refined_T);
  }
  RCLCPP_INFO(get_logger(), "[GicpMapper] Captura incluida fitness=%.4f", score);
  refined_pose_ = refined_T;
 
  local_window_.push_back(aligned);
  if (local_window_.size() > local_window_size_) {
    local_window_.pop_front();
  }


  if (isKeyframe()) {
      addToMap(aligned);
      kf_pose_ = refined_pose_;
      ++kf_count_;
  }
  
  publishAll(msg->header, refined_pose_, aligned);
}

  // ===========================================================================
  CloudTPtr preprocess(const sensor_msgs::msg::PointCloud2 & ros_cloud)
  {
    auto raw = std::make_shared<CloudT>();
    pcl::fromROSMsg(ros_cloud, *raw);

    const float min2 = min_range_ * min_range_;
    const float max2 = max_range_ * max_range_;

    auto filtered = std::make_shared<CloudT>();
    filtered->reserve(raw->size());
    for (const auto & p : *raw) {
      if (!std::isfinite(p.x)) continue;
      const float d2 = p.x*p.x + p.y*p.y + p.z*p.z;
      if (d2 >= min2 && d2 <= max2) filtered->push_back(p);
    }

    auto out = std::make_shared<CloudT>();
    if (!filtered->empty()) {
      voxel_in_.setInputCloud(filtered);
      voxel_in_.filter(*out);
    }
    return out->empty() ? filtered : out;
  }

  // ===========================================================================
  bool isKeyframe()   // llamar con mtx_ tomado
  {
    if (kf_count_ == 0) return true;
    const Eigen::Matrix4f delta = kf_pose_.inverse() * refined_pose_;
    const float dist  = delta.block<3,1>(0,3).norm();
    const float trace = delta(0,0) + delta(1,1) + delta(2,2);
    const float angle = std::acos(std::clamp((trace - 1.f) / 2.f, -1.f, 1.f));
    return dist >= keyframe_dist_ || angle >= keyframe_angle_;
  }

  // ===========================================================================
  void addToMap(const CloudTPtr & cloud)   // llamar con mtx_ tomado
  {
    keyframes_.push_back(cloud);
    if (keyframes_.size() > map_max_frames_) {
      keyframes_.pop_front();
      map_cloud_->clear();
      for (const auto & kf : keyframes_) *map_cloud_ += *kf;
    } else {
      *map_cloud_ += *cloud;
    }
  }

  // ===========================================================================
  void publishAll(const std_msgs::msg::Header & header,
                  const Eigen::Matrix4f        & pose,
                  const CloudTPtr              & aligned)
  {
    if (pub_aligned_->get_subscription_count() > 0) {
      sensor_msgs::msg::PointCloud2 out;
      pcl::toROSMsg(*aligned, out);
      out.header.stamp    = header.stamp;
      out.header.frame_id = "map";
      pub_aligned_->publish(out);
    }

    if (pub_map_->get_subscription_count() > 0 && !map_cloud_->empty()) {
      sensor_msgs::msg::PointCloud2 out;
      pcl::toROSMsg(*map_cloud_, out);
      out.header.stamp    = header.stamp;
      out.header.frame_id = "map";
      pub_map_->publish(out);
    }
    (void)pose;
  }

  // Llamar con mtx_ tomado
  CloudTPtr buildLocalTarget()
  {
    auto target = std::make_shared<CloudT>();
    for (const auto & cloud : local_window_) {
      *target += *cloud;
    }

    // Voxelizar para que GICP no se asfixie con 5x más puntos
    if (target->size() > 10000) {
      auto tmp = std::make_shared<CloudT>();
      voxel_local_window_.setInputCloud(target);
      voxel_local_window_.filter(*tmp);
      return tmp;
    }
    return target;
  }

  // ── ROS ────────────────────────────────────────────────────────────────────
  rclcpp::Subscription<CustomMsg>::SharedPtr                    sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   pub_map_, pub_aligned_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr            srv_save_map_;

  // ── Registradores (shared_ptr, solo uno inicializado según use_vgicp_) ─────
  std::shared_ptr<fast_gicp::FastGICP<PointT,PointT>>  gicp_;
  std::shared_ptr<fast_gicp::FastVGICP<PointT,PointT>> vgicp_;

  // ── Filtros ────────────────────────────────────────────────────────────────
  pcl::VoxelGrid<PointT> voxel_in_, voxel_map_, voxel_local_window_;

  // ── Estado ─────────────────────────────────────────────────────────────────
  bool            initialized_ {false};
  Eigen::Matrix4f refined_pose_, kf_pose_;
  CloudTPtr       last_cloud_;
  CloudTPtr       map_cloud_;
  std::deque<CloudTPtr> keyframes_;
  std::mutex      mtx_;
  std::size_t     kf_count_ {0};

  // ── Parámetros ─────────────────────────────────────────────────────────────
  bool        use_vgicp_, force_accept_;
  double      leaf_size_input_, leaf_size_map_, min_range_, max_range_, vgicp_resolution_;
  double      transf_epsilon_, max_corr_dist_, keyframe_dist_, keyframe_angle_,rot_epsilon_;
  double      score_warn_threshold_, save_voxel_size_;
  int         max_iterations_, num_threads_;
  std::size_t map_max_frames_;
  std::string save_map_path_, file_map_;

  int retry_count;

  // En la sección de miembros privados
  std::deque<CloudTPtr>  local_window_;        // ventana de N nubes en frame mapa
  std::size_t            local_window_size_;
  double                 leaf_size_local_target_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GicpMappingNode>());
  rclcpp::shutdown();
  return 0;
}