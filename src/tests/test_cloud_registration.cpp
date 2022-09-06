#include <iostream>
#include <utils/cloud_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
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
  
  Projector projector = json::loadProjectorConfig(path + LIDAR_CONFIG_FILE);
  SuperPointDetector superpoint = json::loadSuperPointConfig(path, SUPERPOINT_CONFIG_FILE);
  Tracker tracker = json::loadMatchConfig(Matcher::BFMatcher, path + MATCH_CONFIG_FILE);
  auto [ransac_iterations, inliers_threshold] = json::loadRANSACConfig(path + RANSAC_CONFIG_FILE);
  Registrator registrator = Registrator(ransac_iterations, inliers_threshold);
  Image last_img;

  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != CLOUD_TOPIC) continue;
    
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);
    
    vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    superpoint.detectAndCompute(img.intensity(), keypoints, descriptors);

    Pointset2f matches = tracker.update(keypoints, descriptors);

    Pointset3f pointset(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
      pointset[i] = {img.get3DPoint(matches[i].first), img.get3DPoint(matches[i].second)};
    }

    auto [transform, inliers, mask] = registrator.registerPoints(pointset); 

    if (abs(transform(0,0)) > 10.f || abs(transform(1,3)) > 10.f) {
      cout << "Anomalous Transform\n";
      cout << transform.affine() << "\n";
      cout << "point matches = " << pointset.size() << "\ninliers = " << inliers.size() << "\n";
      break;
    } 

    cv::Mat registered_img = Image::drawMatches(last_img, img, matches, mask);
    last_img = img.clone();
    
    cv::imshow("Feature Detection", registered_img);
    cv::waitKey(1);
  }
  
  bag.close();
  return 0;
}