#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/gps_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <utils/matplotlibcpp.hpp>
#include <core/superpoint.hpp>
#include <core/tracker.hpp>
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
  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path+LIDAR_CONFIG_FILE);
  auto [sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  auto [type, knn_threshold, norm_threshold, norm_type] = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  auto [iterations, kernel_threshold, damping, inliers_threshold] = json::loadICPConfig(path + ICP_CONFIG_FILE);
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(-1, sp_threshold, nms_dist, false, path + weights_file);
  Tracker tracker(matcher, knn_threshold, norm_threshold, norm_type);
  Registrator registrator = Registrator(Estimator::ICP, iterations, inliers_threshold, kernel_threshold, damping);

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
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
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