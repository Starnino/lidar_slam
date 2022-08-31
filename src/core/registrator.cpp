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
  Vector3f mu_x = Vector3f::Zero(); Vector3f mu_y = Vector3f::Zero();
  Matrix3f Sigma = Matrix3f::Zero();
  float sigma_x = 0.f; float sigma_y = 0.f;
  int size = points.size();
  
  for (int i = 0; i < size; ++i) {
    mu_x += points[i].first;
    mu_y += points[i].second;
  }
  mu_x /= size; mu_y /= size;

  for (int i = 0; i < size; ++i) {
    Vector3f vx = points[i].first - mu_x;
    Vector3f vy = points[i].second - mu_y;
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
  
  _rotation = R;
  _translation = t;
  _inliers.clear();
}

int Pose::evaluate(Data<Pointset3f>& data, float inliers_threshold) {
  Points* points = static_cast<Points*>(&data);
  int num_inliers = 0;

  for (pair<Point3f,Point3f>& pointpair : points->_points) {
    Point3f transformed_point = predict(pointpair.first);
    Point3f diff = pointpair.second - transformed_point;
    float euclidean_dist = sqrt(diff.dot(diff));

    if (euclidean_dist < inliers_threshold) {
      num_inliers++;
      _inliers.push_back(pointpair);
    }
  }

  return num_inliers;
}

tuple<Affine3f,Pointset3f>  Registrator::registerPoints(Pointset3f& pointset) {
  if (pointset.size() != 0) {
    Points points(pointset);
    Model<Pointset3f>* model = RANSAC<Pose,Pointset3f>(points, _ransac_iterations, _inliers_threshold, 3);
    Pose pose = *(static_cast<Pose*>(model));
    return {pose.matrix(), pose.inliers()};
    //_prev_pose = _prev_pose*pose;
  }

  return {_prev_pose.matrix(), {}};
}

