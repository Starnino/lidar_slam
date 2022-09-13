#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <utils/cloud_helper.hpp>
#include <utils/gps_helper.hpp>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <core/superpoint.hpp>
#include <core/tracker.hpp>
#include <core/registrator.hpp>

using std::cout;

int main(int argc, char **argv) {
  
  ros::init(argc, argv, "lidar_odometry_publisher");
  ros::NodeHandle nh;
  ros::Publisher odom_publisher = nh.advertise<nav_msgs::Odometry>(LIDAR_ODOMETRY_TOPIC, 50);

  string path = ros::package::getPath(PACKAGE_NAME);
  auto [height, width, fov_up, fov_down, max_depth, max_intensity] = json::loadProjectorConfig(path+LIDAR_CONFIG_FILE);
  auto [sp_threshold, nms_dist, weights_file] = json::loadSuperPointConfig(path, DETECTOR_CONFIG_FILE);
  auto [type, knn_threshold, norm_threshold, norm_type] = json::loadMatchConfig(path + MATCH_CONFIG_FILE);
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  auto [iterations, inliers_threshold] = json::loadRANSACConfig(path + RANSAC_CONFIG_FILE);
  //auto [iterations, kernel_threshold, damping, inliers_threshold] = json::loadICPConfig(path + ICP_CONFIG_FILE);
  
  Projector projector(height, width, fov_up, fov_down, max_depth, max_intensity);
  SuperPointDetector detector(-1, sp_threshold, nms_dist, false, path + weights_file);
  Tracker tracker(matcher, knn_threshold, norm_threshold, norm_type);
  Registrator registrator = Registrator(Estimator::RANSAC, iterations, inliers_threshold);

  Pose pose;
  
  auto cloud_handler = [&](const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    PointCloud cloud = deserializeCloudMsg(cloud_msg);
    Image img = pointCloud2Img(cloud, projector);

    vector<cv::KeyPoint> keypoints; cv::Mat descriptors;
    detector.detectAndCompute(img.intensity(), keypoints, descriptors);
    Pointset3f matches = std::get<1>(tracker.update(keypoints, descriptors, img));
    auto [found, transform] = registrator.registerPoints(matches);
    
    if (found) {
      pose = pose*transform;
      Eigen::Quaternionf quaternion(pose.rotation());

      nav_msgs::Odometry odom;
      odom.header.stamp = ros::Time::now();
      odom.header.frame_id = "initial_pose";
      odom.child_frame_id = cloud_msg->header.frame_id;
      odom.pose.pose.position.x = pose.translation().x();
      odom.pose.pose.position.y = pose.translation().y();
      odom.pose.pose.position.z = pose.translation().z();
      odom.pose.pose.orientation.x = quaternion.x();
      odom.pose.pose.orientation.y = quaternion.y();
      odom.pose.pose.orientation.z = quaternion.z();
      odom.pose.pose.orientation.w = quaternion.w();

      odom_publisher.publish(odom);
    }
  };
    
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(CLOUD_TOPIC, 1, cloud_handler);  
  ros::spin();

  return 0;
}