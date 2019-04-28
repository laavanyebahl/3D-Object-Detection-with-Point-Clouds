#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <ar_track_alvar_msgs/AlvarMarkers.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <geometry_msgs/Pose.h>

tf2_ros::Buffer* tf_buffer;
tf2_ros::TransformListener* tf_listener;

ros::Publisher pub;


void cb(ar_track_alvar_msgs::AlvarMarkers req) {


    if (!req.markers.empty()) {
      tf::Quaternion q(req.markers[0].pose.pose.orientation.x, req.markers[0].pose.pose.orientation.y, req.markers[0].pose.pose.orientation.z, req.markers[0].pose.pose.orientation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      ROS_INFO("roll, pitch, yaw=%1.2f  %1.2f  %1.2f", roll, pitch, yaw);
      std::cout<<"Pose fom Camera= "<<req.markers[0].pose.pose<<std::endl;

      geometry_msgs::TransformStamped transformStamped;

      try {
        transformStamped = tf_buffer->lookupTransform("front_base_footprint", "ar_marker_"+ std::to_string(req.markers[0].id), req.header.stamp);
      } catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
        ros::Duration(1.0).sleep();
        return;
      }

      std::cout<<"Pose fom Robot= "<<transformStamped.transform<<std::endl;

      //convert transformStamped to Pose and publish to 'target_pose'
      geometry_msgs::Pose pose;

      pose.position.x = transformStamped.transform.translation.x;
      pose.position.y = transformStamped.transform.translation.y;
      pose.position.z = transformStamped.transform.translation.z;
      pose.orientation.x = transformStamped.transform.rotation.x;
      pose.orientation.y = transformStamped.transform.rotation.y;
      pose.orientation.z = transformStamped.transform.rotation.z;
      pose.orientation.w = transformStamped.transform.rotation.w;


      pub.publish(pose);

    }
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "arlistener");
  ros::NodeHandle nh;
  pub = nh.advertise<geometry_msgs::Pose>("target_pose", 1000);

  tf_buffer = new tf2_ros::Buffer();
  tf_listener = new tf2_ros::TransformListener(*tf_buffer);

  ros::Subscriber sub = nh.subscribe("ar_pose_marker", 1, cb);

  // tf2_ros::Buffer tfBuffer;
  // tf2_ros::TransformListener tfListener(tfBuffer);

  ros::spin();
  return 0;

}
