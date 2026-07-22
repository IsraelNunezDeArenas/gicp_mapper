#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/random_sample.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace gicp_mapping
{

using PointT   = pcl::PointXYZRGB;
using CloudT   = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

// ── Resultado de cada llamada a addCloud ──────────────────────────────────────
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
  CloudPtr        cloud_full;    // nube completa con color → guardado / publicación
  CloudPtr        cloud_sparse;  // nube reducida (N pts fijos) → GICP
};

// ── Parámetros ────────────────────────────────────────────────────────────────
struct MapperParams
{
  int    num_threads             {4};
  int    max_iterations          {64};
  double max_correspondence_dist {2.0};
  double transformation_epsilon  {1e-3};

  // Puntos máximos que se pasan a GICP como source (nube entrante)
  // y como target (mapa local acumulado).
  // Reducir para mayor velocidad; aumentar para mayor precisión.
  int    gicp_source_points      {5000};
  int    gicp_target_points      {20000};

  // Compatibilidad con parámetros anteriores (ignorados internamente)
  double input_leaf_size         {0.0};
  double map_leaf_size           {0.0};

  double keyframe_delta_trans    {0.30};
  double keyframe_delta_angle    {0.15};
  int    local_map_window        {20};
};

// ── GICPMapper ────────────────────────────────────────────────────────────────
class GICPMapper
{
public:
  explicit GICPMapper(const MapperParams & p)
  : params_(p)
  , target_dirty_(false)
  {
    gicp_.setNumThreads(p.num_threads);
    gicp_.setMaximumIterations(p.max_iterations);
    gicp_.setMaxCorrespondenceDistance(p.max_correspondence_dist);
    gicp_.setTransformationEpsilon(p.transformation_epsilon);
  }

  // ── API principal ──────────────────────────────────────────────────────────
  RegistrationResult addCloud(
    const CloudPtr & raw,
    const rclcpp::Time & stamp,
    const std::optional<Eigen::Matrix4f> & external_guess = std::nullopt)
  {
    RegistrationResult res;

    // ── Preprocesado de la nube entrante ──────────────────────────────────────
    // cloud_full : NaN eliminados, color original intacto → se almacena
    // cloud_sparse: submuestra fija para GICP → se descarta tras el frame
    auto cloud_full   = removeNaN(raw);
    if (cloud_full->empty()) return res;
    auto cloud_sparse = randomSample(cloud_full, params_.gicp_source_points);

    std::lock_guard<std::mutex> lk(mutex_);

    // ── Primer frame ──────────────────────────────────────────────────────────
    if (keyframes_.empty()) {
      Eigen::Matrix4f init = external_guess.value_or(Eigen::Matrix4f::Identity());
      createKeyframe(cloud_full, cloud_sparse, stamp, init);
      // Construye el target sparse y lo carga en GICP (solo una vez aquí)
      rebuildSparseTarget();
      updateGICPTarget();
      res.converged   = true;
      res.is_keyframe = true;
      res.pose        = init;
      res.guess_used  = init;
      return res;
    }

    // ── Actualizar target GICP solo si hay un nuevo keyframe pendiente ────────
    // target_dirty_ se activa en createKeyframe(); se desactiva aquí.
    // Así el KD-tree no se recalcula en frames que no generan keyframe.
    if (target_dirty_) {
      rebuildSparseTarget();
      updateGICPTarget();
      target_dirty_ = false;
    }

    // ── Initial guess ─────────────────────────────────────────────────────────
    Eigen::Matrix4f guess = external_guess.value_or(current_pose_);
    res.guess_used = guess;

    // ── GICP ─────────────────────────────────────────────────────────────────
    // El target (KD-tree + covarianzas) ya está precalculado.
    // Solo se recalcula el source en cada frame.
    CloudPtr aligned(new CloudT);
    gicp_.setInputSource(cloud_sparse);
    gicp_.align(*aligned, guess);

    if (!gicp_.hasConverged()) return res;

    res.converged  = true;
    res.pose       = gicp_.getFinalTransformation();
    res.fitness    = gicp_.getFitnessScore();
    last_fitness_  = res.fitness;
    current_pose_  = res.pose;

    // ── Criterio de keyframe ──────────────────────────────────────────────────
    Eigen::Matrix4f delta = keyframes_.back().pose.inverse() * res.pose;
    double dt = delta.block<3,1>(0,3).norm();
    double da = Eigen::AngleAxisf(delta.block<3,3>(0,0)).angle();

    if (dt < params_.keyframe_delta_trans && da < params_.keyframe_delta_angle)
      return res;

    // Nuevo keyframe: marca el target como desactualizado para el próximo frame
    createKeyframe(cloud_full, cloud_sparse, stamp, res.pose);
    res.is_keyframe = true;
    return res;
  }

  // ── Consultas ─────────────────────────────────────────────────────────────

  // Mapa global completo con color original de cada punto.
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

  Eigen::Matrix4f getCurrentPose() const { std::lock_guard<std::mutex> lk(mutex_); return current_pose_; }
  double          getLastFitness() const { return last_fitness_; }
  int             keyframeCount()  const { std::lock_guard<std::mutex> lk(mutex_); return (int)keyframes_.size(); }

private:
  // ── Helpers internos ──────────────────────────────────────────────────────

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
    current_pose_  = pose;
    target_dirty_  = true;   // señala que el target de GICP debe actualizarse
  }

  // Acumula las nubes sparse de la ventana deslizante y las submuestrea
  // a gicp_target_points para mantener el KD-tree acotado.
  // Solo se llama cuando target_dirty_ == true.
  void rebuildSparseTarget()
  {
    auto accum = std::make_shared<CloudT>();
    const int start = std::max(0, (int)keyframes_.size() - params_.local_map_window);
    for (int i = start; i < (int)keyframes_.size(); ++i) {
      CloudT tmp;
      pcl::transformPointCloud(*keyframes_[i].cloud_sparse, tmp, keyframes_[i].pose);
      *accum += tmp;
    }
    // Limitar el target a gicp_target_points para que el KD-tree sea rápido
    local_map_sparse_ = randomSample(accum, params_.gicp_target_points);
  }

  // Carga el target en GICP: precalcula KD-tree y covarianzas una sola vez.
  // Llamar solo tras rebuildSparseTarget().
  void updateGICPTarget()
  {
    gicp_.setInputTarget(local_map_sparse_);
  }

  // Elimina NaN/Inf. Obligatorio antes de KD-tree / GICP.
  static CloudPtr removeNaN(const CloudPtr & in)
  {
    auto out = std::make_shared<CloudT>();
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*in, *out, idx);
    return out;
  }

  // Random sampling con conteo fijo. Si la nube ya es más pequeña, la devuelve tal cual.
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
  MapperParams                           params_;
  fast_gicp::FastGICP<PointT, PointT>   gicp_;

  mutable std::mutex    mutex_;
  std::vector<Keyframe> keyframes_;
  CloudPtr              local_map_sparse_;          // target GICP cacheado
  bool                  target_dirty_;              // ¿necesita reconstruirse?
  Eigen::Matrix4f       current_pose_ {Eigen::Matrix4f::Identity()};
  double                last_fitness_ {0.0};
};

}  // namespace gicp_mapping