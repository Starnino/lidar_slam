#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <utils/cloud_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>

using std::cout;

int main(int argc, char **argv) {
  
  ros::init(argc, argv, "feature_extractor");
  ros::NodeHandle nh;

  string path = ros::package::getPath(PACKAGE_NAME);
  
  Projector projector = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  SuperPointDetector superpoint = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);

  auto cloud_handler = [&](const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);

    vector<cv::KeyPoint> keypoints;
    superpoint.detect(img.intensity(), keypoints);
  };
    
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(CLOUD_TOPIC, 1, cloud_handler);  
  ros::spin();

  return 0;
}

