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
  Affine3f _transform;
  Pointset3f _inliers;
  vector<bool> _mask;
  
  explicit Pose() { 
    _transform = Affine3f::Identity();
  }
  explicit Pose(Affine3f transform) {
    _transform = transform;
  }
  explicit Pose(Matrix3f rotation, Vector3f translation) {
    Affine3f a(rotation);
    a.translation() = translation;
    _transform = a;
  }
  inline const Affine3f& transform() const {return _transform; }
  inline const float transform(int r, int c) const {return _transform(r,c); }
  inline void setTransform(Affine3f transform){ _transform = transform; }
  inline const Vector3f translation() const {return _transform.translation(); }
  inline const Matrix3f rotation() const {return _transform.rotation(); }
  inline const Pointset3f& inliers() const { return _inliers; } 
  inline const vector<bool> mask() const { return _mask; }
  inline Pose operator*(const Pose& other) { return Pose(transform()*other.transform()); }
  inline Point3f operator*(const Point3f& point) { return transform()*point; }
  inline Point3f predict(Point3f& point) { return transform()*point; }

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
      srand(time(NULL));
      _ransac_iterations = ransac_iterations;
      _inliers_threshold = inliers_threshold;
    }

    static Affine3f computeAlignment(const Pointset3f& pointset);
    tuple<bool, Pose> registerPoints(Pointset3f& points);
};