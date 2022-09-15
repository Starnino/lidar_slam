#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <core/superpoint.hpp>
#include <core/tracker.hpp>
#include <core/registrator.hpp>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <opencv2/highgui.hpp>

using std::cout;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO BAG FILE]\n";
    return 1;
  }

  string path = ros::package::getPath(PACKAGE_NAME);
  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  auto [sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  auto [type, knn_threshold, norm_threshold, norm_type] = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  auto [iterations, kernel_threshold, damping, inliers_threshold] = json::loadICPConfig(path + ICP_CONFIG_FILE);
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(-1, sp_threshold, nms_dist, false, path + weights_file);
  Tracker tracker(matcher, knn_threshold, norm_threshold, norm_type);
  Registrator registrator = Registrator(Estimator::ICP, iterations, inliers_threshold, kernel_threshold, damping);
  Image last_img;

  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != CLOUD_TOPIC) continue;
    
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector); 
    vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
    auto [matches2D, matches3D] = tracker.update(keypoints, descriptors, img);
    auto [found, pose] = registrator.registerPoints(matches3D);
    
    if (abs(pose.transform(0,0)) > 10.f || abs(pose.transform(1,3)) > 10.f) {
      cout << "Anomalous Transform\n";
      cout << pose.transform().affine() << "\n";
      cout << "point matches = " << matches3D.size() << "\ninliers = " << pose.inliers().size() << "\n";
      break;
    } 

    cv::Mat registered_img = Image::drawMatches(last_img, img, matches2D, pose.mask());
    last_img = img.clone();
    
    cv::imshow("Feature Detection", registered_img);
    cv::waitKey(1);
  }
  
  bag.close();
  return 0;
}