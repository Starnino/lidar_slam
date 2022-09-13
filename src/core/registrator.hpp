#pragma once 

#include <string>
#include <tuple>
#include <utils/ransac.hpp>
#include <utils/icp.hpp>

using std::string;

enum class Estimator {RANSAC, ICP};

class Registrator {
    
  private:
    Estimator _estimator;
    Pose _prev_pose;
    int _iterations;
    float _inliers_threshold;
    float _kernel_threshold;
    float _damping;

  public:
    inline explicit Registrator(Estimator estimator, int iterations, float inliers_threshold, float kernel_threshold = 0, float damping = 0) {
      srand(time(NULL));
      _estimator = estimator;
      _iterations = iterations;
      _inliers_threshold = inliers_threshold;
      _kernel_threshold = kernel_threshold;
      _damping = damping;
    }
    tuple<bool, Pose> registerPoints(Pointset3f& points);
};