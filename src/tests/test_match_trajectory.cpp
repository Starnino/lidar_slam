#include <iostream>
#include <chrono>
#include <utils/cloud_helper.hpp>
#include <utils/gps_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <utils/matplotlibcpp.hpp>
#include <utils/input_parser.hpp>
#include <utils/gps_helper.hpp>
#include <core/superpoint.hpp>
#include <core/tracker.hpp>
#include <core/registrator.hpp>
#include <ros/package.h>
#include <sensor_msgs/NavSatFix.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

using std::cout;
namespace plt = matplotlibcpp;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO BAG FILE] [OPTIONS]\n";
    cout << "   -save [NAME]            save plot with <name>";
    return 1;
  }
  InputParser input(argc, argv);
  string path = ros::package::getPath(PACKAGE_NAME);

  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  auto [sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  auto [type, knn_threshold, norm_threshold, norm_type] = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  auto [icp_iterations, kernel_threshold, damping, icp_inliers_threshold] = json::loadICPConfig(path + ICP_CONFIG_FILE);
  auto [ransac_iterations, ransac_inliers_threshold] = json::loadRANSACConfig(path + RANSAC_CONFIG_FILE);
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(-1, sp_threshold, nms_dist, false, path + weights_file);
  Tracker icp_tracker(matcher, knn_threshold, norm_threshold, norm_type); 
  Tracker ransac_tracker(matcher, knn_threshold, norm_threshold, norm_type);
  Registrator icp_registrator, ransac_registrator;
  icp_registrator = Registrator(Estimator::ICP, icp_iterations, icp_inliers_threshold, kernel_threshold, damping);
  ransac_registrator = Registrator(Estimator::RANSAC, ransac_iterations, ransac_inliers_threshold);
 
  vector<float> x_lidar_icp, y_lidar_icp, x_lidar_ransac, y_lidar_ransac, x_gps, y_gps;
  Image img; PointCloud cloud; 
  Pointset3f matches; vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
  bool found; Pose transform;
  Pose icp_pose, ransac_pose;

  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    
    if (m.getTopic() != CLOUD_TOPIC) continue;
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>(); 
    
    // ICP
    cloud = deserializeCloudMsg(cloud_msg);
    img = pointCloud2Img(cloud, projector);
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
    matches = std::get<1>(icp_tracker.update(keypoints, descriptors, img));
    std::tie(found, transform) = icp_registrator.registerPoints(matches);

    if (found) {
      icp_pose = icp_pose*transform;
      x_lidar_icp.push_back(icp_pose.translation().x());
      y_lidar_icp.push_back(icp_pose.translation().y());
    }  

    // RANSAC
    cloud = deserializeCloudMsg(cloud_msg);
    img = pointCloud2Img(cloud, projector);
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
    matches = std::get<1>(ransac_tracker.update(keypoints, descriptors, img));
    std::tie(found, transform) = ransac_registrator.registerPoints(matches);

    if (found) {
      ransac_pose = ransac_pose*transform;
      x_lidar_ransac.push_back(ransac_pose.translation().x());
      y_lidar_ransac.push_back(ransac_pose.translation().y());
    }  
  }
  bag.close();

  // plot trajectories
  plt::plot(x_lidar_icp, y_lidar_icp, "tab:red", {{"label", "lidar-icp"}});
  plt::plot(x_lidar_ransac, y_lidar_ransac, "tab:green", {{"label", "lidar-ransac"}});
  plt::legend();
  plt::title("Trajectory");
  plt::xlabel("x [m]");
  plt::ylabel("y [m]");
  if (input.cmdOptionExists("-save")) {
    plt::savefig(path + "/plots/" + input.getCmdOption("-save") + ".pdf");
    cout << "file " << input.getCmdOption("-save") << ".pdf saved in " << path + "/plots\n";
  }
  plt::show();

  return 0;
}