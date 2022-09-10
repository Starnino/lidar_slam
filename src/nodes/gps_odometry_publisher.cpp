#include <iostream>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <utils/gps_helper.hpp>
#include <utils/define.hpp>

using std::cout;

int main(int argc, char **argv) {
  
  ros::init(argc, argv, "gps_odometry_publisher");
  ros::NodeHandle nh;
  ros::Publisher odom_publisher = nh.advertise<nav_msgs::Odometry>(GPS_ODOMETRY_TOPIC, 50);
  
  double initial_x = 0.0;
  double initial_y = 0.0;
  double initial_z = 0.0;
  bool first = true;

  auto gps_handler = [&](const sensor_msgs::NavSatFix::ConstPtr& gps_msg) {
    double latitude = gps_msg->latitude;
    double longitude = gps_msg->longitude;
    auto [x, y, z] = geodetic2ecef(latitude, longitude);

    if (first) {
      initial_x = x;
      initial_y = y;
      initial_z = z;
      first = false;
    }
    
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = "initial_pose";
    odom.pose.pose.position.x = x - initial_x;
    odom.pose.pose.position.y = y - initial_y;
    odom.pose.pose.position.z = z - initial_z;
    odom.pose.pose.orientation.x = 0;
    odom.pose.pose.orientation.y = 0;
    odom.pose.pose.orientation.z = 0;
    odom.pose.pose.orientation.w = 0;
    odom.child_frame_id = gps_msg->header.frame_id;

    odom_publisher.publish(odom);
  };
    
  ros::Subscriber sub = nh.subscribe<sensor_msgs::NavSatFix>(GPS_TOPIC, 1, gps_handler);  
  ros::spin();

  return 0;
}