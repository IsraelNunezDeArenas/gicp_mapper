#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <segmentation_msgs/msg/semantic_point_cloud.hpp>
#include "gicp_mapper/msg/ndt_cloud.hpp"
#include "gicp_mapper/msg/ndt_cell.hpp"
#include "ndt_cell.hpp"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <bonxai.hpp>

using PointT     = pcl::PointXYZRGB;
using CloudT     = pcl::PointCloud<PointT>;
using CustomMsg  = segmentation_msgs::msg::SemanticPointCloud;
using CustomPtr  = CustomMsg::SharedPtr;

using NDTCell = gicp_mapper::NDTCell;


class NDTVoxelizer : public rclcpp::Node
{

    public:

    NDTVoxelizer(): Node("ndt_voxelizer"){

        resolution_ = declare_parameter("resolution", 0.5);

        scan_grid_ = std::make_unique<Bonxai::VoxelGrid<NDTCell>>(resolution_,4,4);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10000));
        qos.reliable();

        sub_ = create_subscription<CustomMsg>(
            "/cloud_in",
            qos,
            std::bind(&NDTVoxelizer::scan_callback,this,std::placeholders::_1));

        ndt_pub_ = create_publisher<gicp_mapper::msg::NDTCloud>(
            "/ndt_cloud",
            qos);
    }

    private:

    void scan_callback(const CustomPtr msg)
    {

        scan_grid_ = std::make_unique<Bonxai::VoxelGrid<NDTCell>>(resolution_,4,4);

        CloudT::Ptr cloud(new CloudT);
        pcl::fromROSMsg(msg->cloud, *cloud);

        auto accessor = scan_grid_->createAccessor();

        for(const auto& p : cloud->points)
        {
            if(!std::isfinite(p.x) ||
               !std::isfinite(p.y) ||
               !std::isfinite(p.z))
                continue;
        
            Bonxai::CoordT coord = scan_grid_->posToCoord(p.x,p.y,p.z);
        
            NDTCell* cell = accessor.value(coord,true);
        
            Eigen::Vector3d pt(p.x,p.y,p.z);
        
            cell->sum += pt;
            cell->sum_outer += pt*pt.transpose();

            cell->sum_r += static_cast<double>(p.b);    //BGR -> RGB
            cell->sum_g += static_cast<double>(p.g);
            cell->sum_b += static_cast<double>(p.r);

            cell->num_points++;
        }

        computeStatistics();

        publish(msg);

    }

    void computeStatistics()
    {
        scan_grid_->forEachCell(
            [&](NDTCell& cell, const Bonxai::CoordT&)
            {
                if(cell.num_points < 4)
                {
                    cell.valid = false;
                    return;
                }


                const double n =
                    static_cast<double>(cell.num_points);


                cell.mean =
                    cell.sum / n;


                cell.covariance =
                    cell.sum_outer / n -
                    cell.mean * cell.mean.transpose();


                // Regularización para evitar matrices singulares
                cell.covariance +=
                    Eigen::Matrix3d::Identity() * 1e-3;

                //Promediado del color
                cell.r = static_cast<uint8_t>(std::round(cell.sum_r / n));
                cell.g = static_cast<uint8_t>(std::round(cell.sum_g / n));
                cell.b = static_cast<uint8_t>(std::round(cell.sum_b / n));


                cell.valid = true;
            });
    }


    void publish(const CustomPtr& msg)
{
    gicp_mapper::msg::NDTCloud cloud_msg;


    cloud_msg.header = msg->header;

    // Si Custom tiene pose
    cloud_msg.pose = msg->pose;

    cloud_msg.resolution = resolution_;


    scan_grid_->forEachCell(
        [&](NDTCell& cell, const Bonxai::CoordT& coord)
            {
                if(!cell.valid)
                    return;


                gicp_mapper::msg::NDTCell ndt_cell;


                // Coordenada del voxel
                ndt_cell.voxel_x = coord.x;
                ndt_cell.voxel_y = coord.y;
                ndt_cell.voxel_z = coord.z;


                // -----------------------------
                // Sumatorio de puntos
                // -----------------------------
                ndt_cell.sum.x = cell.sum.x();
                ndt_cell.sum.y = cell.sum.y();
                ndt_cell.sum.z = cell.sum.z();



                // -----------------------------
                // Sum outer product
                // -----------------------------
                for(int i=0;i<3;i++)
                {
                    for(int j=0;j<3;j++)
                    {
                        ndt_cell.sum_outer[i*3+j] =
                            cell.sum_outer(i,j);
                    }
                }


                // -----------------------------
                // Media
                // -----------------------------
                ndt_cell.mean_x =
                    cell.mean.x();

                ndt_cell.mean_y =
                    cell.mean.y();

                ndt_cell.mean_z =
                    cell.mean.z();



                // -----------------------------
                // Covarianza
                // -----------------------------
                for(int i=0;i<3;i++)
                {
                    for(int j=0;j<3;j++)
                    {
                        ndt_cell.covariance[i*3+j] =
                            cell.covariance(i,j);
                    }
                }

                // -----------------------------
                // Color
                // -----------------------------

                ndt_cell.r = cell.r;
                ndt_cell.g = cell.g;
                ndt_cell.b = cell.b;

                // -----------------------------
                // Número puntos
                // -----------------------------
                ndt_cell.num_points =
                    cell.num_points;


                ndt_cell.valid =
                    cell.valid;


                cloud_msg.cells.push_back(ndt_cell);

            });


        cloud_msg.num_points =
            cloud_msg.cells.size();


        ndt_pub_->publish(cloud_msg);
    }



    double resolution_;
    std::unique_ptr<Bonxai::VoxelGrid<NDTCell>> scan_grid_;

    rclcpp::Subscription<CustomMsg>::SharedPtr sub_;

    rclcpp::Publisher<gicp_mapper::msg::NDTCloud>::SharedPtr ndt_pub_;


    };

int main(int argc, char **argv)
    {
        rclcpp::init(argc, argv);

        rclcpp::spin(
            std::make_shared<NDTVoxelizer>());

        rclcpp::shutdown();

        return 0;
    }