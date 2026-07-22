#pragma once

#include <Eigen/Dense>
#include <cstdint>


namespace gicp_mapper
{

struct NDTCell
{

    // Acumuladores de color (para promediar)
    double sum_r = 0.0;
    double sum_g = 0.0;
    double sum_b = 0.0;

    // Color final (calculado en computeStatistics)
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;


    Eigen::Vector3d sum = Eigen::Vector3d::Zero();

    Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();

    uint32_t num_points = 0;

    bool valid = false;
};

}