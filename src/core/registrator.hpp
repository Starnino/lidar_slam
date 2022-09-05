#pragma once 

#include <tuple>
#include <core/image.hpp>
#include <utils/ransac.hpp>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using std::vector;
using std::pair;
using std::tuple;
using Eigen::Affine3f;
using Eigen::Vector3f;
using Eigen::Matrix3f;
using Eigen::JacobiSVD;
using Point3f = Vector3f;
using Pointset3f = vector<pair<Point3f,Point3f>>;

struct Points : Data<Pointset3f> {
  Pointset3f _points;

  Points() = default;
  explicit Points(Pointset3f& points) { _points = points; }

  Pointset3f randomSelect(int n) override;
};

struct Pose : Model<Pointset3f> {
  Matrix3f _rotation;
  Vector3f _translation;
  Pointset3f _inliers;
  vector<bool> _mask;
  
  explicit Pose() { 
    _rotation = Matrix3f::Identity();
    _translation = Vector3f::Zero();
  }
  explicit Pose(Matrix3f rotation, Vector3f translation) {
    _rotation = rotation;
    _translation = translation;
  }
  inline const Matrix3f& rotation() const { return _rotation; }
  inline const Vector3f& translation() const { return _translation; }
  inline const Pointset3f& inliers() const { return _inliers; }
  inline const vector<bool> mask() const { return _mask; }
  inline Pose operator*(const Pose& other) {
      Matrix3f R = rotation()*other.rotation();
      Vector3f t = rotation()*other.translation() + translation();
      return Pose(R,t); 
  }
  inline Affine3f matrix() {
    Affine3f matrix(_rotation);
    matrix.translation() = _translation;
    return matrix;
  }
  inline Point3f predict(Point3f& point) { return rotation()*point + translation(); }

  void fit(Pointset3f& points) override;
  int evaluate(Data<Pointset3f>& points, float inlier_threshold) override;
  inline Pose* clone() const { return new Pose(*this); }
};

class Registrator {
    
  private:
    int _ransac_iterations;
    float _inliers_threshold;
    Pose _prev_pose; 

  public:
    inline explicit Registrator(int ransac_iterations, float inliers_threshold) {
      srand(0);
      _ransac_iterations = ransac_iterations;
      _inliers_threshold = inliers_threshold;
      _prev_pose = Pose();
    }

    tuple<Affine3f,Pointset3f,vector<bool>> registerPoints(Pointset3f& points);
};