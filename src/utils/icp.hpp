#pragma once

#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <tuple>
#include <core/pose.hpp>

using std::pair;
using std::tuple;
using std::vector;
using Eigen::ArrayXf;
using Eigen::Affine3f;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::AngleAxisf;
using Eigen::Quaternionf;
using Eigen::Matrix;
using Matrix6f = Matrix<float,6,6>;
using Matrix3x6f = Matrix<float,3,6>;
using Vector6f = Matrix<float,6,1>;
using Point3f = Vector3f;
using Pointset3f = vector<pair<Point3f,Point3f>>;

Affine3f vector2transform(Vector6f v);
Matrix3f skew(Vector3f v);
tuple<Vector3f,Matrix3x6f> errorAndJacobian(Affine3f pose, Point3f point, Point3f measurements);
tuple<Affine3f,vector<float>> ICP(Pointset3f points, int iterations, float kernel_threshold, float damping, Affine3f guess = Affine3f::Identity());