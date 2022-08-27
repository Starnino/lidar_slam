#pragma once

#include <Eigen/Core>

using Eigen::ArrayXf;

class PointCloud {

  private:
    ArrayXf _x;
    ArrayXf _y;
    ArrayXf _z;
    ArrayXf _i;
    int _w;
    int _h;
    
  public:
    explicit PointCloud(int height, int width);
        
    inline const ArrayXf& x() { return _x; }
    inline const ArrayXf& y() { return _y; }
    inline const ArrayXf& z() { return _z; }
    inline const ArrayXf& i() { return _i; }
        
    inline const int& h() { return _h; }
    inline const int& w() { return _w; }

    inline float& x(int index) { return _x(index); }
    inline float& y(int index) { return _y(index); }
    inline float& z(int index) { return _z(index); }
    inline float& i(int index) { return _i(index); }

    struct Point3DI {
      float x;
      float y;
      float z;
      float i;
    };

    const Point3DI p(int index);
};