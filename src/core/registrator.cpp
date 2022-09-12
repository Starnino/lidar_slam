#include "registrator.hpp"

Pointset3f Points::randomSelect(int n) {
  int size = _points.size();
  int index = rand() % size;
  int stride = rand() % (size-n+1) + 1;

  Pointset3f random_points(n);
  for (int i = 0; i < n; ++i) {
    random_points[i] = _points[index];
    index = (index + stride) % size;
  }
  return random_points;
}

void Pose::fit(Pointset3f& points) {
  Affine3f sol = Registrator::computeAlignment(points);
  _transform = sol;
  _inliers.clear();
  _mask.clear();
}

int Pose::evaluate(Data<Pointset3f>& data, float inliers_threshold) {
  Points* points = static_cast<Points*>(&data);
  int num_inliers = 0;

  for (pair<Point3f,Point3f>& pointpair : points->_points) {
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

Affine3f Registrator::computeAlignment(const Pointset3f& pointset) {
  Vector3f mu_x = Vector3f::Zero(); Vector3f mu_y = Vector3f::Zero();
  Matrix3f Sigma = Matrix3f::Zero();
  float sigma_x = 0.f; float sigma_y = 0.f;
  int size = pointset.size();
  
  for (int i = 0; i < size; ++i) {
    mu_x += pointset[i].first;
    mu_y += pointset[i].second;
  }
  mu_x /= size; mu_y /= size;

  for (int i = 0; i < size; ++i) {
    Vector3f vx = pointset[i].first - mu_x;
    Vector3f vy = pointset[i].second - mu_y;
    sigma_x += vx.transpose()*vx;
    sigma_y += vy.transpose()*vy;
    Sigma += vy*vx.transpose(); 
  }
  Sigma /= size; sigma_x /= size; sigma_y /= size;

  JacobiSVD<Matrix3f> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Matrix3f S = Matrix3f::Identity();
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) {
    S = Eigen::DiagonalMatrix<float, 3>(1, 1, -1);
  }

  Matrix3f R = svd.matrixU() * S * svd.matrixV().transpose();
  Vector3f t = mu_y - R * mu_x;

  Affine3f sol(R);
  sol.translation() = t;
  return sol;
}

tuple<bool,Pose> Registrator::registerPoints(Pointset3f& pointset) {
  if (pointset.size() != 0) {
    Points points(pointset);
    auto [found, model] = RANSAC<Pose,Pointset3f>(points, _ransac_iterations, _inliers_threshold, 3);
    if (found) {
      Pose pose = *(static_cast<Pose*>(model));
      pose.setTransform(computeAlignment(pose.inliers()));
      return {true, pose};
    }
  }
  return {false, Pose()};
}