// Author: Laavanye Bahl

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

/*
* This node transforms point cloud from /camera_link frame to /world frame
*/

class CloudTransformer
{
public:
  explicit CloudTransformer(ros::NodeHandle nh): nh_(nh)  {
    // Define Publishers and Subscribers here
    pcl_sub_ = nh_.subscribe("/camera/depth/color/points", 1, &CloudTransformer::pclCallback, this);
    pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/front_base_footprint/points", 1);

    buffer_.reset(new sensor_msgs::PointCloud2);
    buffer_->header.frame_id = "front_base_footprint";
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber pcl_sub_;
  ros::Publisher pcl_pub_;
  tf::TransformListener listener_;
  sensor_msgs::PointCloud2::Ptr buffer_;

  void pclCallback(const sensor_msgs::PointCloud2ConstPtr& pcl_msg)
  {
    listener_.waitForTransform("/front_base_footprint", "/camera_depth_optical_frame", ros::Time::now(), ros::Duration(1.0));
    pcl_ros::transformPointCloud("/front_base_footprint", *pcl_msg, *buffer_, listener_);
    pcl_pub_.publish(buffer_);
  }
};  // End of class CloudTransformer

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_tf");
  ros::NodeHandle nh;

  CloudTransformer tranform_cloud(nh);

  // Spin until ROS is shutdown
  while (ros::ok())
    ros::spin();

  return 0;
}
