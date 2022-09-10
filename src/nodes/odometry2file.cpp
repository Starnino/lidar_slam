#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <utils/define.hpp>

using std::cout;
using std::string;
using namespace message_filters;
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry,nav_msgs::Odometry> ApproximateTimePolicy;

std::ofstream trajectory_file;

void callback(const nav_msgs::Odometry& gps_odom,
              const nav_msgs::Odometry& lidar_odom) {
  trajectory_file << gps_odom.pose.pose.position.x << ","
                  << gps_odom.pose.pose.position.y << ","
                  << gps_odom.pose.pose.position.z << ",";
  trajectory_file << lidar_odom.pose.pose.position.x << ","
                  << lidar_odom.pose.pose.position.y << ","
                  << lidar_odom.pose.pose.position.z;
  trajectory_file << std::endl;
};

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: odometry2file [FILENAME]\n";
    return 1;
  }

  ros::init(argc, argv, "odometry2file");
  ros::NodeHandle nh;
  
  trajectory_file.open(argv[1]);
  Subscriber<nav_msgs::Odometry> gps_sub(nh, GPS_ODOMETRY_TOPIC, 1);
  Subscriber<nav_msgs::Odometry> lidar_sub(nh, LIDAR_ODOMETRY_TOPIC, 1);
  Synchronizer<ApproximateTimePolicy> sync(ApproximateTimePolicy(10), gps_sub, lidar_sub);
  sync.registerCallback(callback);

  ros::spin();
  trajectory_file.close();
  return 0;
}