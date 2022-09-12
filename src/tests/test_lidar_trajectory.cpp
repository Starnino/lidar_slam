#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/gps_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <utils/matplotlibcpp.hpp>
#include <core/registrator.hpp>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

using std::cout;
namespace plt = matplotlibcpp;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO BAG FILE]\n";
    return 1;
  }

  string path = ros::package::getPath(PACKAGE_NAME);
  
  Projector projector = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  SuperPointDetector superpoint = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  Tracker tracker = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  auto [ransac_iterations, inliers_threshold] = json::loadRANSACConfig(path + RANSAC_CONFIG_FILE);
  Registrator registrator = Registrator(ransac_iterations, inliers_threshold);

  Pose pose;
  vector<float> x_lidar; vector<float> y_lidar;
  
  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != CLOUD_TOPIC) continue;
    
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);
    
    cv::Mat mat; vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    cv::cvtColor(img.intensity(), mat, cv::COLOR_GRAY2RGB);
    superpoint.detectAndCompute(img.intensity(), keypoints, descriptors);
    Pointset3f matches = std::get<1>(tracker.update(keypoints, descriptors, img));
    auto [found, transform] = registrator.registerPoints(matches);
    
    if (found) {
      pose = pose*transform;
      x_lidar.push_back(pose.translation().x());
      y_lidar.push_back(pose.translation().y());
    }
  }
  bag.close();

  // plot trajectories
  plt::plot(x_lidar, y_lidar, "tab:red");
  plt::title("Lidar Trajectory");
  plt::xlabel("x [m]");
  plt::ylabel("y [m]"); 
  plt::show();

  return 0;
}