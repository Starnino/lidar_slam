#include "ransac.hpp"

Pointset3f randomSelect(Pointset3f& points, int n) {
  int size = points.size();
  int index = rand() % size;
  int stride = rand() % (size-n+1) + 1;

  Pointset3f random_points(n);
  for (int i = 0; i < n; ++i) {
    random_points[i] = points[index];
    index = (index + stride) % size;
  }
  return random_points;
}

Affine3f computeAlignment(const Pointset3f& pointset) {
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

tuple<bool,Pose> RANSAC(Pointset3f& data, int num_iterations, float inliers_threshold, int n) {
  Pose best_model; Pose model;
  int best_score = 0; int iteration = 0;

  while (iteration++ < num_iterations) {
    Pointset3f random_data = randomSelect(data, n);
    model.setTransform(computeAlignment(random_data));
    int score = model.evaluate(data, inliers_threshold);
    
    if (score > best_score) {
      best_model = model.clone();
      best_score = score;
    }
  }
  if (best_score == 0) {
    return {false, Pose()};
  }
  else return {true, best_model};
}