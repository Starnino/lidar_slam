#include <iostream>
#include <ros/package.h>
#include <utils/json_helper.cpp>
#include <utils/define.hpp>
#include <utils/icp.hpp>

using std::cout;

int main(int argc, char **argv) {
  srand(time(NULL));
  string path = ros::package::getPath(PACKAGE_NAME);
  auto [iterations, kernel_threshold, damping, inliers_threshold] = json::loadICPConfig(path + ICP_CONFIG_FILE);

  // random rotation matrix
  Matrix3f R = Eigen::Quaternionf::UnitRandom().toRotationMatrix();
  Vector3f t{1, 2, 0};
  Affine3f gt_pose(R);
  gt_pose.translation() = t;
  cout << "Ground Thruth pose\n";
  cout << gt_pose.affine() << "\n\n";

  Pointset3f set;
  // generate random points with correct association (addiong some noise)
  for (int i = 0; i < 50; ++i) {
    Point3f point = Point3f::Random()*10;
    set.push_back({point, gt_pose*point  + Point3f::Random()*0.05f});
  }
  // generate random points with wrong associations
  for (int i = 0; i < 50; ++i) {
    set.push_back({Point3f::Random()*10, Point3f::Random()*10});
  }

  auto [transform, chi] = ICP(set, iterations, kernel_threshold, damping);
  Pose pose(transform);
  pose.evaluate(set, inliers_threshold);

  cout << "Estimated pose\n";
  cout << pose.transform().affine() << "\n";
  cout << "\nNumber of inliers = " << pose.inliers().size() << "\n";
  
  return 0;
}