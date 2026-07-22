#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <segmentation_msgs/msg/semantic_point_cloud.hpp>

#include <Eigen/Core>
#include <mutex>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::PoseWithCovariance & pwc)
{
  const auto & p = pwc.pose;

  Eigen::Quaternionf q(
    p.orientation.w,
    p.orientation.x,
    p.orientation.y,
    p.orientation.z
  );
  q.normalize();

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3,3>(0,0) = q.toRotationMatrix();

  T(0,3) = p.position.x;
  T(1,3) = p.position.y;
  T(2,3) = p.position.z;

  return T;
}

class VGICPMapperNode : public rclcpp::Node
{
public:
  VGICPMapperNode()
  : Node("vgicp_mapper_node")
  {
    declare_parameter("input_topic", "/points");
    declare_parameter("voxel_resolution", 0.3);
    declare_parameter("keyframe_dist", 0.5);

    voxel_res_ = get_parameter("voxel_resolution").as_double();
    keyframe_dist_ = get_parameter("keyframe_dist").as_double();

    sub_ = create_subscription<segmentation_msgs::msg::SemanticPointCloud>(
      get_parameter("input_topic").as_string(),
      10,
      std::bind(&VGICPMapperNode::callback, this, std::placeholders::_1)
    );

    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/vgicp_map", 1);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/vgicp_pose", 10);

    vgicp_.setMaxCorrespondenceDistance(1.0);
    vgicp_.setResolution(voxel_res_);
    vgicp_.setMaximumIterations(50);
    vgicp_.setNumThreads(8);

    global_map_.reset(new CloudT);
    current_pose_ = Eigen::Matrix4f::Identity();

    RCLCPP_INFO(get_logger(), "VGICP Mapper iniciado");
  }

private:
  void callback(const segmentation_msgs::msg::SemanticPointCloud::SharedPtr msg)
{
  // ─────────────────────────────────────────────
  // 1. Convertir nube
  // ─────────────────────────────────────────────
  CloudT::Ptr cloud(new CloudT);
  pcl::fromROSMsg(msg->cloud, *cloud);

  if (cloud->empty()) {
    RCLCPP_WARN(get_logger(), "Nube vacía");
    return;
  }

  // ─────────────────────────────────────────────
  // 2. Inicialización del mapa
  // ─────────────────────────────────────────────
  if (global_map_->empty()) {
    *global_map_ = *cloud;
    vgicp_.setInputTarget(global_map_);
    current_pose_ = Eigen::Matrix4f::Identity();

    RCLCPP_INFO(get_logger(),
      "Mapa inicializado con %zu puntos",
      global_map_->size());

    return;
  }

  // IMPORTANTE: asegurar target válido SIEMPRE
  if (!global_map_ || global_map_->empty()) {
    RCLCPP_ERROR(get_logger(), "ERROR: mapa inválido");
    return;
  }

  vgicp_.setInputTarget(global_map_);

  // ─────────────────────────────────────────────
  // 3. Guess inicial (validado)
  // ─────────────────────────────────────────────
  Eigen::Matrix4f guess = poseToMatrix(msg->pose);

  if (!std::isfinite(guess(0,3))) {
    guess = current_pose_;  // fallback seguro
  }

  // ─────────────────────────────────────────────
  // 4. VGICP align
  // ─────────────────────────────────────────────
  vgicp_.setInputSource(cloud);

  CloudT aligned;
  auto t1 = now();

  vgicp_.align(aligned, guess);

  auto t2 = now();

  RCLCPP_INFO(get_logger(),
    "VGICP time: %.3f ms",
    (t2 - t1).seconds() * 1000.0);

  // ─────────────────────────────────────────────
  // 5. Verificación de convergencia REAL
  // ─────────────────────────────────────────────
  if (!vgicp_.hasConverged()) {
    RCLCPP_WARN(get_logger(), "VGICP no convergió");

    RCLCPP_ERROR(get_logger(),
      "Fitness: %.6f",
      vgicp_.getFitnessScore());

    RCLCPP_ERROR(get_logger(),
      "Target size: %zu | Source size: %zu",
      global_map_->size(), cloud->size());

    return;
  }

  double fitness = vgicp_.getFitnessScore();

  Eigen::Matrix4f T = vgicp_.getFinalTransformation();

  // ─────────────────────────────────────────────
  // 6. Update pose
  // ─────────────────────────────────────────────
  float dist = (T.block<3,1>(0,3) -
                current_pose_.block<3,1>(0,3)).norm();

  current_pose_ = T;

  // ─────────────────────────────────────────────
  // 7. Keyframe logic MEJORADO
  // ─────────────────────────────────────────────
  const float fitness_threshold = 0.05;  // ajustable

  if (dist > keyframe_dist_ && fitness < fitness_threshold)
  {
    CloudT transformed;
    pcl::transformPointCloud(*cloud, transformed, current_pose_);

    *global_map_ += transformed;

    // Downsample seguro
    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(0.1f, 0.1f, 0.1f);
    vg.setInputCloud(global_map_);

    CloudT tmp;
    vg.filter(tmp);

    if (!tmp.empty()) {
      *global_map_ = tmp;
      vgicp_.setInputTarget(global_map_);
    }

    RCLCPP_INFO(get_logger(),
      "Keyframe añadido | pts=%zu | fitness=%.6f",
      global_map_->size(), fitness);
  }

  // ─────────────────────────────────────────────
  // 8. Publicación
  // ─────────────────────────────────────────────
  publishPose(msg->header.stamp);
  publishMap(msg->header.stamp);
}

  void publishPose(const rclcpp::Time& stamp)
  {
    geometry_msgs::msg::PoseStamped pose;

    pose.header.stamp = stamp;
    pose.header.frame_id = "map";

    Eigen::Matrix3f R = current_pose_.block<3,3>(0,0);
    Eigen::Quaternionf q(R);

    pose.pose.position.x = current_pose_(0,3);
    pose.pose.position.y = current_pose_(1,3);
    pose.pose.position.z = current_pose_(2,3);

    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();

    pose_pub_->publish(pose);
  }

  void publishMap(const rclcpp::Time& stamp)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*global_map_, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = "map";
    map_pub_->publish(msg);
  }

  rclcpp::Subscription<segmentation_msgs::msg::SemanticPointCloud>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  fast_gicp::FastVGICP<PointT, PointT> vgicp_;

  CloudT::Ptr global_map_;
  Eigen::Matrix4f current_pose_;

  double voxel_res_;
  double keyframe_dist_;

  std::mutex mutex_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VGICPMapperNode>());
  rclcpp::shutdown();
  return 0;
}