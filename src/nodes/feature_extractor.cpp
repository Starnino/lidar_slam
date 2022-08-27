#include "ros/ros.h"
#include <iostream>
#include <sensor_msgs/PointCloud2.h>

#define TOPIC "/os1_cloud_node/points"

using std::cout;
using std::endl;

int main(int argc, char **argv) {
  
  ros::init(argc, argv, "feature_extractor");
  ros::NodeHandle nh;

  auto cloud_handler = [&](const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    cout << cloud_msg->point_step << endl;
  };
    
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(TOPIC, 1, cloud_handler);  
  ros::spin();

  return 0;
}

