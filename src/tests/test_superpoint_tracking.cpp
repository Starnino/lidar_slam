#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/json_helper.cpp>
#include <core/superpoint.hpp>
#include <core/tracker.hpp>
#include <utils/define.hpp>
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
  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path+LIDAR_CONFIG_FILE);
  auto [sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  auto [type, knn_threshold, norm_threshold, norm_type] = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(-1, sp_threshold, nms_dist, false, path + weights_file);
  Tracker tracker(matcher, knn_threshold, norm_threshold, norm_type);
  
  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != CLOUD_TOPIC) continue;
    
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);

    vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
    auto tracks = tracker.updateTracks(keypoints, descriptors);
    
    img.drawTracks(tracks);  
    cv::imshow("Feature Detection", img.intensity());
    cv::waitKey(1);
  }
  
  bag.close();
  return 0;
}