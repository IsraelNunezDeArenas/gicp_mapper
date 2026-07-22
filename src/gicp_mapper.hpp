#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <rclcpp/rclcpp.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace gicp_mapping
{

using PointT   = pcl::PointXYZ;
using CloudT   = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

// ── Keyframe ──────────────────────────────────────────────────────────────────
struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int             id;
  rclcpp::Time    stamp;
  Eigen::Matrix4f pose;   // pose acumulada en el frame del mapa
  CloudPtr        cloud;  // nube en frame local (sensor / base_link)
};

// ── Parámetros ────────────────────────────────────────────────────────────────
struct MapperParams
{
  int    num_threads            {4};
  int    max_iterations         {64};
  double max_correspondence_dist{2.0};
  double transformation_epsilon {1e-3};

  double input_leaf_size        {0.1};    // downsampling nube entrada (m)
  double map_leaf_size          {0.15};   // downsampling mapa/local (m)

  double keyframe_delta_trans   {0.3};    // distancia mínima entre keyframes (m)
  double keyframe_delta_angle   {0.15};   // ángulo mínimo entre keyframes (rad)

  int    local_map_window       {20};     // nº keyframes en ventana deslizante
};

// ── GICPMapper ────────────────────────────────────────────────────────────────
class GICPMapper
{
public:
  explicit GICPMapper(const MapperParams & p)
  : params_(p)
  {
    gicp_.setNumThreads(p.num_threads);
    gicp_.setMaximumIterations(p.max_iterations);
    gicp_.setMaxCorrespondenceDistance(p.max_correspondence_dist);
    gicp_.setTransformationEpsilon(p.transformation_epsilon);
  }

  // Procesa nube nueva. Devuelve true si se creó un keyframe.
  bool addCloud(const CloudPtr & raw, const rclcpp::Time & stamp)
  {
    auto cloud = downsample(raw, params_.input_leaf_size);
    if (cloud->empty()) return false;

    std::lock_guard<std::mutex> lk(mutex_);

    if (keyframes_.empty()) {
      createKeyframe(cloud, stamp, Eigen::Matrix4f::Identity());
      rebuildLocalMap();
      return true;
    }

    // Registrar contra mapa local deslizante
    CloudPtr aligned(new CloudT);
    gicp_.setInputTarget(local_map_);
    gicp_.setInputSource(cloud);
    gicp_.align(*aligned, current_pose_);

    if (!gicp_.hasConverged()) return false;

    Eigen::Matrix4f new_pose = gicp_.getFinalTransformation();
    last_fitness_ = gicp_.getFitnessScore();

    // Criterio de keyframe: sólo si nos movimos suficiente
    Eigen::Matrix4f delta = current_pose_.inverse() * new_pose;
    double dt = delta.block<3,1>(0,3).norm();
    double da = Eigen::AngleAxisf(delta.block<3,3>(0,0)).angle();

    current_pose_ = new_pose;   // actualizar pose siempre

    if (dt < params_.keyframe_delta_trans && da < params_.keyframe_delta_angle) {
      return false;
    }

    createKeyframe(cloud, stamp, new_pose);
    rebuildLocalMap();
    return true;
  }

  // Mapa global: todas las nubes transformadas al frame de mapa
  CloudPtr buildGlobalMap() const
  {
    std::lock_guard<std::mutex> lk(mutex_);
    auto global = std::make_shared<CloudT>();
    for (const auto & kf : keyframes_) {
      CloudT tmp;
      pcl::transformPointCloud(*kf.cloud, tmp, kf.pose);
      *global += tmp;
    }
    return downsample(global, params_.map_leaf_size);
  }

  CloudPtr         getLocalMap()    const { std::lock_guard<std::mutex> lk(mutex_); return local_map_; }
  Eigen::Matrix4f  getCurrentPose() const { std::lock_guard<std::mutex> lk(mutex_); return current_pose_; }
  double           getLastFitness() const { return last_fitness_; }
  int              keyframeCount()  const { std::lock_guard<std::mutex> lk(mutex_); return (int)keyframes_.size(); }

private:
  void createKeyframe(const CloudPtr & cloud,
                      const rclcpp::Time & stamp,
                      const Eigen::Matrix4f & pose)
  {
    Keyframe kf;
    kf.id    = (int)keyframes_.size();
    kf.stamp = stamp;
    kf.pose  = pose;
    kf.cloud = cloud;
    keyframes_.push_back(std::move(kf));
    current_pose_ = pose;
  }

  void rebuildLocalMap()
  {
    local_map_ = std::make_shared<CloudT>();
    int start = std::max(0, (int)keyframes_.size() - params_.local_map_window);
    for (int i = start; i < (int)keyframes_.size(); ++i) {
      CloudT tmp;
      pcl::transformPointCloud(*keyframes_[i].cloud, tmp, keyframes_[i].pose);
      *local_map_ += tmp;
    }
    local_map_ = downsample(local_map_, params_.map_leaf_size);
  }

  static CloudPtr downsample(const CloudPtr & in, double leaf)
  {
    if (leaf <= 0.0) return std::make_shared<CloudT>(*in);
    auto out = std::make_shared<CloudT>();
    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(float(leaf), float(leaf), float(leaf));
    vg.setInputCloud(in);
    vg.filter(*out);
    return out;
  }

  MapperParams                       params_;
  fast_gicp::FastGICP<PointT, PointT> gicp_;

  mutable std::mutex       mutex_;
  std::vector<Keyframe>    keyframes_;
  CloudPtr                 local_map_;
  Eigen::Matrix4f          current_pose_ {Eigen::Matrix4f::Identity()};
  double                   last_fitness_ {0.0};
};

}  // namespace gicp_mapping
