#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/json_helper.cpp>
#include <core/tracker.hpp>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <opencv2/highgui.hpp>

#define TOPIC "/os1_cloud_node/points"
#define PACKAGE_NAME "lidarslam"
#define LIDAR_CONFIG_FILE "/config/lidar.cfg"
#define SUPERPOINT_CONFIG_FILE "/config/superpoint.cfg"
#define MATCH_CONFIG_FILE "/config/match.cfg"

using std::cout;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO BAG FILE]\n";
    return 1;
  }
  
  string path = ros::package::getPath(PACKAGE_NAME);
  
  Projector projector = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  SuperPointDetector superpoint = json::loadSuperPointConfig(path, SUPERPOINT_CONFIG_FILE);
  Tracker tracker = json::loadMatchConfig(Matcher::BFMatcher, MATCH_CONFIG_FILE);

  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != TOPIC) continue;
    
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);

    vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    superpoint.detectAndCompute(img.intensity(), keypoints, descriptors);

    auto tracks = tracker.updateTracks(keypoints, descriptors);
    
    Image detection(cloud.h(), cloud.w());
    cv::cvtColor(img.intensity(), detection.intensity(), cv::COLOR_GRAY2RGB);
    detection.drawTracks(tracks);
  
    cv::imshow("Feature Detection", detection.intensity());
    cv::waitKey(1);
  }
  
  bag.close();
  return 0;
}