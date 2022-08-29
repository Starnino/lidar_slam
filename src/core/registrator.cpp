#include "registrator.hpp"

Pointset3f Points::randomSelect(int n) {
  int size = _points.size();
  srand(time(NULL));
  int index = rand() % size;
  int stride = rand() % size + 1;

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
    Sigma += (points[i].second - mu_y)*(points[i].first - mu_x).transpose();
  }
  Sigma /= size; sigma_x /= size; sigma_y /= size;

  JacobiSVD<Matrix3f> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Matrix3f S = Matrix3f::Identity();
  if (Sigma.determinant() < 0) {
    S = Eigen::DiagonalMatrix<float, 3>(1, 1, -1);
  }

  Matrix3f R = svd.matrixU() * S * svd.matrixV().transpose();
  Vector3f t = mu_y - R * mu_x;
  
  _rotation = R;
  _translation = t;
  _inliers.clear();
}

int Pose::evaluate(Data<Pointset3f>& data, float inlier_threshold) {
  Points* points = static_cast<Points*>(&data);
  int num_inliers = 0;

  for (pair<Point3f,Point3f>& pointpair : points->_points) {
    Point3f transformed_point = predict(pointpair.first);
    Point3f diff = pointpair.second - transformed_point;
    float euclidean_dist = sqrt(diff.dot(diff));

    if (euclidean_dist < inlier_threshold) {
      num_inliers++;
      _inliers.push_back(pointpair);
    }
  }

  return num_inliers;
}

Registrator::Registrator(int ransac_iterations, float inlier_threshold) {
  _ransac_iterations = ransac_iterations;
  _inlier_threshold = inlier_threshold;
}

Pose Registrator::registerPoints(Pointset3f& points) {
  if (true){

  }
  return Pose();
}

