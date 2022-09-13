#pragma once

#include <core/pose.hpp>
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

Pointset3f randomSelect(Pointset3f& points, int n);
Affine3f computeAlignment(const Pointset3f& points);
tuple<bool,Pose> RANSAC(Pointset3f& data, int num_iterations, float inliers_threshold, int n);