#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/ply_io.h>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace gicp_mapping
{

using PointT   = pcl::PointXYZRGB;
using CloudT   = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

struct RegistrationResult
{
  bool            converged   {false};
  bool            is_keyframe {false};
  Eigen::Matrix4f pose        {Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f guess_used  {Eigen::Matrix4f::Identity()};
  double          fitness     {0.0};
};

struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int             id;
  rclcpp::Time    stamp;
  Eigen::Matrix4f pose;
  CloudPtr        cloud_full;    // nube completa con color → voxel map global
  CloudPtr        cloud_sparse;  // nube reducida            → source VGICP
};

struct MapperParams
{
  int    num_threads             {4};
  int    max_iterations          {64};
  double max_correspondence_dist {2.0};
  double transformation_epsilon  {1e-3};

  double vgicp_resolution        {0.3};   // vóxel interno de VGICP (matching)
  int    source_max_points       {5000};  // puntos source por frame

  // Resolución del voxel grid del mapa global guardado.
  // Un punto por vóxel: cuanto mayor, menos puntos en el mapa final.
  // Independiente de vgicp_resolution.
  double map_voxel_size          {0.05};  // metros

  int    local_map_window        {20};
  double keyframe_delta_trans    {0.30};
  double keyframe_delta_angle    {0.15};

  double input_leaf_size         {0.0};   // ignorado (compatibilidad)
  double map_leaf_size           {0.0};   // ignorado (compatibilidad)
};

// ── Voxel grid incremental ────────────────────────────────────────────────────
//
// Tabla hash: clave de vóxel → punto representativo con color original.
// Al insertar un punto en un vóxel ya ocupado se promedia el color RGB
// y se actualiza el centroide XYZ, manteniendo UN solo punto por celda.
//
// Ventajas frente a reconstruir con VoxelGrid en cada guardado:
//   - O(1) por punto al insertar (amortizado)
//   - El mapa se puede guardar en cualquier momento sin reconstruir
//   - El número de puntos es estrictamente ≤ número de vóxeles ocupados
//
struct VoxelCell
{
  float x_sum {0}, y_sum {0}, z_sum {0};
  float r_sum {0}, g_sum {0}, b_sum {0};
  uint32_t count {0};

  void add(const PointT & p)
  {
    x_sum += p.x; y_sum += p.y; z_sum += p.z;
    r_sum += p.r; g_sum += p.g; b_sum += p.b;
    ++count;
  }

  PointT centroid() const
  {
    PointT out;
    float inv = 1.0f / count;
    out.x = x_sum * inv;
    out.y = y_sum * inv;
    out.z = z_sum * inv;
    out.r = static_cast<uint8_t>(std::min(255.0f, r_sum * inv));
    out.g = static_cast<uint8_t>(std::min(255.0f, g_sum * inv));
    out.b = static_cast<uint8_t>(std::min(255.0f, b_sum * inv));
    return out;
  }
};

class VoxelGrid
{
public:
  explicit VoxelGrid(double leaf) : leaf_(leaf), inv_leaf_(1.0 / leaf) {}

  // Inserta todos los puntos de una nube (ya transformada al frame del mapa).
  void insertCloud(const CloudT & cloud)
  {
    for (const auto & p : cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        continue;
      const uint64_t key = voxelKey(p);
      cells_[key].add(p);
    }
  }

  // Extrae el mapa como nube PCL (un punto por vóxel).
  CloudPtr toCloud() const
  {
    auto out = std::make_shared<CloudT>();
    out->reserve(cells_.size());
    for (const auto & [key, cell] : cells_)
      out->push_back(cell.centroid());
    return out;
  }

  std::size_t size() const { return cells_.size(); }

private:
  uint64_t voxelKey(const PointT & p) const
  {
    const auto ix = static_cast<int64_t>(std::floor(p.x * inv_leaf_));
    const auto iy = static_cast<int64_t>(std::floor(p.y * inv_leaf_));
    const auto iz = static_cast<int64_t>(std::floor(p.z * inv_leaf_));
    // Mezcla con primos de Knuth para minimizar colisiones
    return static_cast<uint64_t>(ix) * 2654435761ULL
         ^ static_cast<uint64_t>(iy) * 805459861ULL
         ^ static_cast<uint64_t>(iz) * 3674653429ULL;
  }

  double inv_leaf_;
  double leaf_;
  std::unordered_map<uint64_t, VoxelCell> cells_;
};

// ── GICPMapper ────────────────────────────────────────────────────────────────
class GICPMapper
{
public:
  explicit GICPMapper(const MapperParams & p)
  : params_(p)
  , target_dirty_(false)
  , global_voxels_(p.vgicp_resolution)
  {
    vgicp_.setNumThreads(p.num_threads);
    vgicp_.setMaximumIterations(p.max_iterations);
    vgicp_.setMaxCorrespondenceDistance(p.max_correspondence_dist);
    vgicp_.setTransformationEpsilon(p.transformation_epsilon);
    vgicp_.setResolution(p.vgicp_resolution);
  }

  RegistrationResult addCloud(
    const CloudPtr & raw,
    const rclcpp::Time & stamp,
    const std::optional<Eigen::Matrix4f> & external_guess = std::nullopt)
  {
    RegistrationResult res;

    auto cloud_full   = removeNaN(raw);
    if (cloud_full->empty()) return res;
    auto cloud_sparse = randomSample(cloud_full, params_.source_max_points);

    std::lock_guard<std::mutex> lk(mutex_);

    if (keyframes_.empty()) {
      Eigen::Matrix4f init = external_guess.value_or(Eigen::Matrix4f::Identity());
      createKeyframe(cloud_full, cloud_sparse, stamp, init);
      rebuildTarget();
      res.converged   = true;
      res.is_keyframe = true;
      res.pose        = init;
      res.guess_used  = init;
      return res;
    }

    if (target_dirty_) {
      rebuildTarget();
      target_dirty_ = false;
    }

    Eigen::Matrix4f guess = external_guess.value_or(current_pose_);
    res.guess_used = guess;

    CloudPtr aligned(new CloudT);
    vgicp_.setInputSource(cloud_sparse);
    vgicp_.align(*aligned, guess);

    if (!vgicp_.hasConverged()) return res;

    res.converged  = true;
    res.pose       = vgicp_.getFinalTransformation();
    res.fitness    = vgicp_.getFitnessScore();
    last_fitness_  = res.fitness;
    current_pose_  = res.pose;

    Eigen::Matrix4f delta = keyframes_.back().pose.inverse() * res.pose;
    double dt = delta.block<3,1>(0,3).norm();
    double da = Eigen::AngleAxisf(delta.block<3,3>(0,0)).angle();

    if (dt < params_.keyframe_delta_trans && da < params_.keyframe_delta_angle)
      return res;

    createKeyframe(cloud_full, cloud_sparse, stamp, res.pose);
    res.is_keyframe = true;
    return res;
  }

  // ── Consultas ─────────────────────────────────────────────────────────────

  // Mapa global voxelizado: un punto por vóxel con color promediado.
  // Se construye de forma incremental al añadir keyframes → O(1) aquí.
  CloudPtr buildGlobalMap() const
  {
    std::lock_guard<std::mutex> lk(mutex_);
    return global_voxels_.toCloud();
  }

  // Número de vóxeles ocupados en el mapa global.
  std::size_t globalVoxelCount() const
  {
    std::lock_guard<std::mutex> lk(mutex_);
    return global_voxels_.size();
  }

  // Mapa local (ventana deslizante) con color, para publicación en RViz.
  CloudPtr getLocalMap() const
  {
    std::lock_guard<std::mutex> lk(mutex_);
    auto local = std::make_shared<CloudT>();
    const int start = std::max(0, (int)keyframes_.size() - params_.local_map_window);
    for (int i = start; i < (int)keyframes_.size(); ++i) {
      CloudT tmp;
      pcl::transformPointCloud(*keyframes_[i].cloud_full, tmp, keyframes_[i].pose);
      *local += tmp;
    }
    return local;
  }

  // Guarda el mapa global voxelizado como PLY binario XYZRGB.
  bool saveMapPLY(const std::string & path) const
  {
    auto map = buildGlobalMap();
    if (map->empty()) return false;
    return pcl::io::savePLYFileBinary(path, *map) == 0;
  }

  Eigen::Matrix4f getCurrentPose() const { std::lock_guard<std::mutex> lk(mutex_); return current_pose_; }
  double          getLastFitness() const { return last_fitness_; }
  int             keyframeCount()  const { std::lock_guard<std::mutex> lk(mutex_); return (int)keyframes_.size(); }

private:
  // Al crear un keyframe, sus puntos (transformados al frame del mapa)
  // se insertan inmediatamente en el voxel grid global.
  // Coste: O(N) una sola vez por keyframe.
  void createKeyframe(const CloudPtr & full, const CloudPtr & sparse,
                      const rclcpp::Time & stamp, const Eigen::Matrix4f & pose)
  {
    Keyframe kf;
    kf.id           = (int)keyframes_.size();
    kf.stamp        = stamp;
    kf.pose         = pose;
    kf.cloud_full   = full;
    kf.cloud_sparse = sparse;
    keyframes_.push_back(std::move(kf));
    current_pose_ = pose;
    target_dirty_ = true;

    // Insertar la nube completa en el voxel grid global
    CloudT transformed;
    pcl::transformPointCloud(*full, transformed, pose);
    global_voxels_.insertCloud(transformed);
  }

  void rebuildTarget()
  {
    auto accum = std::make_shared<CloudT>();
    const int start = std::max(0, (int)keyframes_.size() - params_.local_map_window);
    for (int i = start; i < (int)keyframes_.size(); ++i) {
      CloudT tmp;
      pcl::transformPointCloud(*keyframes_[i].cloud_sparse, tmp, keyframes_[i].pose);
      *accum += tmp;
    }
    vgicp_.setInputTarget(accum);
  }

  static CloudPtr removeNaN(const CloudPtr & in)
  {
    auto out = std::make_shared<CloudT>();
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*in, *out, idx);
    return out;
  }

  static CloudPtr randomSample(const CloudPtr & in, int max_pts)
  {
    if (max_pts <= 0 || (int)in->size() <= max_pts)
      return std::make_shared<CloudT>(*in);
    auto out = std::make_shared<CloudT>();
    pcl::RandomSample<PointT> rs;
    rs.setSample(static_cast<unsigned int>(max_pts));
    rs.setSeed(42);
    rs.setInputCloud(in);
    rs.filter(*out);
    return out;
  }

  MapperParams                          params_;
  fast_gicp::FastVGICP<PointT, PointT> vgicp_;

  mutable std::mutex    mutex_;
  std::vector<Keyframe> keyframes_;
  bool                  target_dirty_;
  VoxelGrid             global_voxels_;   // mapa global voxelizado incremental
  Eigen::Matrix4f       current_pose_ {Eigen::Matrix4f::Identity()};
  double                last_fitness_ {0.0};
};

}  // namespace gicp_mapping