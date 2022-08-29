#include <iostream>
#include <ros/package.h>
#include <utils/json_helper.cpp>
#include <core/registrator.hpp>

#define PACKAGE_NAME "lidarslam"
#define RANSAC_CONFIG_FILE "/config/ransac.cfg"

using std::cout;

int main(int argc, char **argv) {
  
  string path = ros::package::getPath(PACKAGE_NAME);
  auto [ransac_iterations, inliers_threshold] = json::loadRANSACConfig(path + RANSAC_CONFIG_FILE);

  // random rotation matrix
  Matrix3f R = Eigen::Quaternionf::UnitRandom().toRotationMatrix();
  Vector3f t{1, 2, 0};
  Pose gt_pose = Pose(R,t);
  cout << "Ground Thruth pose\n";
  cout << gt_pose.matrix().affine() << "\n\n";

  Pointset3f set;
  // generate random points with correct association
  for (int i = 0; i < 25; ++i) {
    Point3f point = Point3f::Random();
    set.push_back({point, R*point + t});
  }
  // generate random points with wrong associations
  for (int i = 0; i < 5; ++i) {
    set.push_back({Point3f::Random(), Point3f::Random()});
  }
  Points points(set);

  Pose model;
  Model<Pointset3f>* ransac = RANSAC<Pointset3f>(model, points, ransac_iterations, inliers_threshold, 3);
  Pose* pose = static_cast<Pose*>(ransac);
  
  cout << "Estimated pose\n";
  cout << pose->matrix().affine() << "\n";
  cout << "Number of inliers = " << pose->inliers().size() << "\n";
  return 0;
}