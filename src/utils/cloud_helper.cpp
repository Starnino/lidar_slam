#include "cloud_helper.hpp"

PointCloud deserializeCloudMsg(const sensor_msgs::PointCloud2ConstPtr msg) {
  int height = msg->height;
  int width = msg->width;
  int point_step = msg->point_step;
  int x_offset = msg->fields[0].offset;
  int y_offset = msg->fields[1].offset;
  int z_offset = msg->fields[2].offset;
  int i_offset = msg->fields[3].offset;

  PointCloud cloud(height, width);

  const unsigned char* point_ptr = msg->data.data();
  #pragma omp parallel for 
  for (int n = 0; n < height*width; ++n) {
    int point_offset = n * point_step;
    cloud.x(n) = *(reinterpret_cast<const float*>(point_ptr + point_offset + x_offset));
    cloud.y(n) = *(reinterpret_cast<const float*>(point_ptr + point_offset + y_offset));
    cloud.z(n) = *(reinterpret_cast<const float*>(point_ptr + point_offset + z_offset));
    cloud.i(n) = *(reinterpret_cast<const float*>(point_ptr + point_offset + i_offset));
  }
  return cloud;
}

template <typename T>
Eigen::Array<T,-1,1> clamp(Eigen::Array<T,-1,1>& arr, T min, T max) {
    arr = arr.unaryExpr([&](const T x) { return std::max<T>(min, x); });
    arr = arr.unaryExpr([&](const T x) { return std::min<T>(max, x); });
    return arr;
};

Image pointCloud2Img(PointCloud& cloud, Projector& projector) {
  int height = projector.height();
  int width = projector.width();
  float fov_up = (projector.fov_up() * EIGEN_PI) / 180.f;
  float fov_down = (projector.fov_down() * EIGEN_PI) / 180.f;
  float max_intensity = projector.max_intensity();
  float max_depth = projector.max_depth();

  // inizialize image
  Image img(height, width);
  
  // spherical projection
  ArrayXf radius = Eigen::sqrt(cloud.x().square() + cloud.y().square() + cloud.z().square());
  ArrayXf azimuth = cloud.y().binaryExpr(cloud.x(), [&](const float y, const float x) { return std::atan2(y,x); });
  ArrayXf elevation = Eigen::asin(cloud.z() / radius);

  // image projection
  ArrayXi u = (float(width) / 2.f * (1.f + azimuth / EIGEN_PI)).round().cast<int>();
  ArrayXi v = (float(height) * (fov_up - elevation) / (fov_up - fov_down))
    .unaryExpr([&](const float x) { return std::isfinite(x)? x : -1.f; }).round().cast<int>();
  
  u = clamp(u, 0, width-1);
  v = clamp(v, 0, height-1);

  // store depth and intensity into img pixels
  ArrayXf norm_i = cloud.i() / max_intensity * 1.f;
  ArrayXf norm_d = 1.f - radius / max_depth * 1.f;
  
  #pragma omp parallel for
  for (int i = 0; i < height*width; ++i) {
    
    int ui = u(i); int vi = v(i); int index = ((vi * width) + ui)*3;
    float intensity = norm_i(i); float depth = norm_d(i);
    
    float* imat_ptr = &(img.intensity().at<float>(vi,ui));
    float* dmat_ptr = &(img.depth().at<float>(vi,ui));
    
    // assign point with smallest range
    if (*imat_ptr == -1.f || *dmat_ptr > depth) {
      *imat_ptr = intensity;
      *dmat_ptr = depth;

      // store 3D point
      img.xyz(index) = cloud.x(i);
      img.xyz(index+1) = cloud.y(i);
      img.xyz(index+2) = cloud.z(i); 
    }
  }

  return img;
}