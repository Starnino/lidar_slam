#include "pose.hpp"

Pose::Pose() { 
  _transform = Affine3f::Identity();
}
Pose::Pose(Affine3f transform) {
  _transform = transform;
}
Pose::Pose(Matrix3f rotation, Vector3f translation) {
  Affine3f a(rotation);
  a.translation() = translation;
  _transform = a;
}

int Pose::evaluate(Pointset3f& points, float inliers_threshold) {
  _inliers.clear();
  _mask.clear();
  int num_inliers = 0;

  for (pair<Point3f,Point3f>& pointpair : points) {
    Point3f diff = pointpair.second - predict(pointpair.first);
    float euclidean_dist = sqrt(diff.dot(diff));

    if (euclidean_dist < inliers_threshold) {
      num_inliers++;
      _inliers.push_back(pointpair);
      _mask.push_back(true);
    }
    else _mask.push_back(false);
  }

  return num_inliers;
}