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
  Affine3f _matrix;
  Pointset3f _inliers;
  vector<bool> _mask;
  
  explicit Pose() { 
    _matrix = Affine3f::Identity();
  }
  explicit Pose(Affine3f matrix) {
    _matrix = matrix;
  }
  explicit Pose(Matrix3f rotation, Vector3f translation) {
    Affine3f a(rotation);
    a.translation() = translation;
    _matrix = a;
  }
  inline const Affine3f matrix() const {return _matrix;}
  inline const Pointset3f& inliers() const { return _inliers; }
  inline const vector<bool> mask() const { return _mask; }
  inline Pose operator*(const Pose& other) { return Pose(matrix()*other.matrix()); }
  inline Point3f operator*(const Point3f& point) { return matrix()*point; }
  inline Point3f predict(Point3f& point) { return matrix()*point; }

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
    tuple<bool, Affine3f> registerPoints(Pointset3f& points);
    tuple<Affine3f,Pointset3f,vector<bool>> registerPoints2(Pointset3f& points);
};