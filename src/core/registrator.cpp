#include "registrator.hpp"

tuple<bool,Pose> Registrator::registerPoints(Pointset3f& points) {
  if (points.size() != 0) {

    if (_estimator == Estimator::RANSAC) {
      auto [found, pose] = RANSAC(points, _iterations, _inliers_threshold, 3);
      if (found) {
        pose.setTransform(computeAlignment(pose.inliers()));
        _prev_pose = pose;  
        return {true, pose};
      }
    }

    else if (_estimator == Estimator::ICP) {
      auto [transform, chi] = ICP(points, _iterations, _kernel_threshold, _damping, _prev_pose.transform());
      Pose pose(transform);
      int score = pose.evaluate(points, _inliers_threshold);
      if (score > 0) {
        _prev_pose = pose;
        return {true, pose};
      }
    }
    
    else throw std::invalid_argument("Specified wrong method!");
  }

  return {false, Pose()};
}