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

namespace gicp_mapping
{

// ── Tipos ─────────────────────────────────────────────────────────────────────
// PointXYZRGB: FastVGICP usa solo XYZ para el matching; RGB viaja gratis.
using PointT   = pcl::PointXYZRGB;
using CloudT   = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

// ── Resultado de addCloud ─────────────────────────────────────────────────────
struct RegistrationResult
{
  bool            converged   {false};
  bool            is_keyframe {false};
  Eigen::Matrix4f pose        {Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f guess_used  {Eigen::Matrix4f::Identity()};
  double          fitness     {0.0};
};

// ── Keyframe ──────────────────────────────────────────────────────────────────
struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int             id;
  rclcpp::Time    stamp;
  Eigen::Matrix4f pose;
  CloudPtr        cloud_full;    // nube completa con color → mapa final
  CloudPtr        cloud_sparse;  // nube reducida            → source VGICP
};

// ── Parámetros ────────────────────────────────────────────────────────────────
struct MapperParams
{
  int    num_threads             {4};
  int    max_iterations          {64};
  double max_correspondence_dist {2.0};
  double transformation_epsilon  {1e-3};

  // Resolución del vóxel interno de VGICP (target).
  // Controla la precisión del matching y la velocidad del KD-tree.
  // Rango típico: 0.1–1.0 m. Valor más pequeño = más preciso y más lento.
  double vgicp_resolution        {0.3};

  // Puntos del source que se pasan a VGICP por frame.
  // VGICP voxeliza el target internamente; el source sí se muestrea aquí.
  int    source_max_points       {5000};

  // Ventana de keyframes para el mapa local (target de VGICP).
  int    local_map_window        {20};

  // Umbrales de nuevo keyframe
  double keyframe_delta_trans    {0.30};
  double keyframe_delta_angle    {0.15};

  // Compatibilidad con versiones anteriores (ignorados)
  double input_leaf_size         {0.0};
  double map_leaf_size           {0.0};
};

// ── GICPMapper ────────────────────────────────────────────────────────────────
class GICPMapper
{
public:
  explicit GICPMapper(const MapperParams & p)
  : params_(p)
  , target_dirty_(false)
  {
    vgicp_.setNumThreads(p.num_threads);
    vgicp_.setMaximumIterations(p.max_iterations);
    vgicp_.setMaxCorrespondenceDistance(p.max_correspondence_dist);
    vgicp_.setTransformationEpsilon(p.transformation_epsilon);
    // Resolución del vóxel del target: VGICP construye internamente
    // un mapa de vóxeles con distribuciones gaussianas.
    vgicp_.setResolution(p.vgicp_resolution);
  }

  // ── API principal ──────────────────────────────────────────────────────────
  RegistrationResult addCloud(
    const CloudPtr & raw,
    const rclcpp::Time & stamp,
    const std::optional<Eigen::Matrix4f> & external_guess = std::nullopt)
  {
    RegistrationResult res;

    auto cloud_full   = removeNaN(raw);
    if (cloud_full->empty()) return res;

    // Source para VGICP: muestreo fijo para velocidad constante por frame
    auto cloud_sparse = randomSample(cloud_full, params_.source_max_points);

    std::lock_guard<std::mutex> lk(mutex_);

    // ── Primer frame ──────────────────────────────────────────────────────────
    if (keyframes_.empty()) {
      Eigen::Matrix4f init = external_guess.value_or(Eigen::Matrix4f::Identity());
      createKeyframe(cloud_full, cloud_sparse, stamp, init);
      rebuildTarget();        // construye voxel map del primer keyframe
      res.converged   = true;
      res.is_keyframe = true;
      res.pose        = init;
      res.guess_used  = init;
      return res;
    }

    // ── Actualizar target VGICP solo cuando hay nuevo keyframe ────────────────
    // El voxel map del target se precalcula en rebuildTarget() y se reutiliza
    // para todos los frames hasta el siguiente keyframe → coste amortizado.
    if (target_dirty_) {
      rebuildTarget();
      target_dirty_ = false;
    }

    // ── Initial guess ─────────────────────────────────────────────────────────
    Eigen::Matrix4f guess = external_guess.value_or(current_pose_);
    res.guess_used = guess;

    // ── VGICP ─────────────────────────────────────────────────────────────────
    // El target (voxel map + distribuciones gaussianas) ya está cargado.
    // Solo se actualiza el source por frame.
    CloudPtr aligned(new CloudT);
    vgicp_.setInputSource(cloud_sparse);
    vgicp_.align(*aligned, guess);

    if (!vgicp_.hasConverged()) return res;

    res.converged  = true;
    res.pose       = vgicp_.getFinalTransformation();
    res.fitness    = vgicp_.getFitnessScore();
    last_fitness_  = res.fitness;
    current_pose_  = res.pose;

    // ── Criterio de keyframe ──────────────────────────────────────────────────
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

  // Mapa global completo: todos los keyframes con color transformados al frame del mapa.
  CloudPtr buildGlobalMap() const
  {
    std::lock_guard<std::mutex> lk(mutex_);
    auto global = std::make_shared<CloudT>();
    for (const auto & kf : keyframes_) {
      CloudT tmp;
      pcl::transformPointCloud(*kf.cloud_full, tmp, kf.pose);
      *global += tmp;
    }
    return global;
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

  // Guarda el mapa global como PLY binario con color XYZRGB.
  // Devuelve true en éxito.
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
  // ── Helpers ───────────────────────────────────────────────────────────────

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
  }

  // Acumula las nubes sparse de la ventana y carga el voxel map en VGICP.
  // VGICP::setInputTarget() construye internamente las distribuciones
  // gaussianas por vóxel → operación costosa, pero solo ocurre aquí.
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

  // ── Miembros ──────────────────────────────────────────────────────────────
  MapperParams                             params_;
  fast_gicp::FastVGICP<PointT, PointT>    vgicp_;

  mutable std::mutex    mutex_;
  std::vector<Keyframe> keyframes_;
  bool                  target_dirty_;
  Eigen::Matrix4f       current_pose_ {Eigen::Matrix4f::Identity()};
  double                last_fitness_ {0.0};
};

}  // namespace gicp_mapping