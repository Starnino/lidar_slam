#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <core/superpoint.hpp>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using std::cout;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO BAG FILE]\n";
    return 1;
  }
  
  string path = ros::package::getPath(PACKAGE_NAME);
  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  auto [nfeatures, sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(nfeatures, sp_threshold, nms_dist, false, path + weights_file);
  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != CLOUD_TOPIC) continue;

    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);
    vector<cv::KeyPoint> keypoints;
    detector.detect(img.intensity(), keypoints);

    img.drawKeypoints(keypoints);
    cv::imshow("Feature Detection", img.intensity());  
    cv::waitKey(1);
  }
  
  bag.close();
  return 0;
}