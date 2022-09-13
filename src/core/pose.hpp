#pragma once

#include <Eigen/Geometry>

using std::vector;
using std::pair;
using Eigen::Affine3f;
using Eigen::Vector3f;
using Eigen::Matrix3f;
using Point3f = Vector3f;
using Pointset3f = vector<pair<Point3f,Point3f>>;

class Pose  {

  private:
    Affine3f _transform;
    Pointset3f _inliers;
    vector<bool> _mask;
  
  public:
    explicit Pose();
    explicit Pose(Affine3f transform);
    explicit Pose(Matrix3f rotation, Vector3f translation);

    inline const Affine3f& transform() const {return _transform; }
    inline const float transform(int r, int c) const {return _transform(r,c); }
    inline void setTransform(Affine3f transform){ _transform = transform; }
    inline const Vector3f translation() const {return _transform.translation(); }
    inline const Matrix3f rotation() const {return _transform.rotation(); }
    inline const Pointset3f& inliers() const { return _inliers; } 
    inline const vector<bool> mask() const { return _mask; }
    inline Point3f predict(Point3f& point) { return transform()*point; }
    inline Pose operator*(const Pose& other) { return Pose(transform()*other.transform()); }
    inline Point3f operator*(const Point3f& point) { return transform()*point; }
    inline Pose clone() const { 
      Pose pose;
      pose._transform = _transform;
      pose._inliers = _inliers;
      pose._mask = _mask;
      return pose; 
    }
    int evaluate(Pointset3f& points, float inlier_threshold);
};
