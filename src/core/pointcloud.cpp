#include "pointcloud.hpp"

PointCloud::PointCloud(int height, int width) { 
  _x = ArrayXf::Zero(height*width);
  _y = ArrayXf::Zero(height*width);
  _z = ArrayXf::Zero(height*width);
  _i = ArrayXf::Zero(height*width);
  _h = height;
  _w = width;
}

const PointCloud::Point3DI PointCloud::p(int index) {
  if (index < 0 || index > x().size()) return Point3DI();
  Point3DI point = {x(index), y(index), z(index), i(index)};
  return point;
} 