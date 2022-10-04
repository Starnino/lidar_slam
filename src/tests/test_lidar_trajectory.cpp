#include <iostream>
#include <chrono>
#include <utils/cloud_helper.hpp>
#include <utils/gps_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <utils/matplotlibcpp.hpp>
#include <utils/input_parser.hpp>
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
    cout << "Usage: lidar_projection [PATH TO BAG FILE] [OPTION]\n";
    cout << "   -estimator   choose estimator among 'icp' or 'ransac'. [default: icp]";
    return 1;
  }
  string path = ros::package::getPath(PACKAGE_NAME);
  
  InputParser input(argc, argv);
  Estimator estimator;
  int iterations; float inliers_threshold; float kernel_threshold = 0.f; float damping = 0.f;
  if (input.cmdOptionExists("-estimator") && input.getCmdOption("-estimator") == "ransac") {
    estimator = Estimator::RANSAC;
    std::tie(iterations, inliers_threshold) = json::loadRANSACConfig(path + RANSAC_CONFIG_FILE);
  }
  else {
    estimator = Estimator::ICP;
    std::tie(iterations, kernel_threshold, damping, inliers_threshold) = json::loadICPConfig(path + ICP_CONFIG_FILE);
  }
  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  auto [nfeatures, sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  auto [type, knn_threshold, norm_threshold, norm_type] = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(nfeatures, sp_threshold, nms_dist, false, path + weights_file);
  Tracker tracker(matcher, knn_threshold, norm_threshold, norm_type);
  Registrator registrator = Registrator(estimator, iterations, inliers_threshold, kernel_threshold, damping);

  Pose pose;
  vector<float> x_lidar; vector<float> y_lidar;
  double ms = 0.0; double count = 0.0;
  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != CLOUD_TOPIC) continue;
    count++;
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    
    auto start = std::chrono::steady_clock::now();
    
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);
    vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
    Pointset3f matches = std::get<1>(tracker.update(keypoints, descriptors, img));
    auto [found, transform] = registrator.registerPoints(matches);
    
    auto end = std::chrono::steady_clock::now();

    if (found) {
      pose = pose*transform;
      x_lidar.push_back(pose.translation().x());
      y_lidar.push_back(pose.translation().y());
    }
    
    ms += std::chrono::duration<double, std::milli>(end-start).count();
  }
  bag.close();

  // frequency
  cout << "data processed at " << (count/ms)*1e3 << " Hz\n";

  // plot trajectories
  plt::plot(x_lidar, y_lidar, "tab:red");
  plt::title("Lidar Trajectory");
  plt::xlabel("x [m]");
  plt::ylabel("y [m]"); 
  plt::show();

  return 0;
}