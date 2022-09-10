#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

#define MIN_CV_32F 0.f
#define MAX_CV_32F 1.f
#define INV_CV_32F -0.1f

using std::vector;
using std::pair;
using std::unordered_map;
using Eigen::ArrayXf;
using Eigen::Vector3f;
using Pointset2f = vector<pair<cv::Point2f,cv::Point2f>>;
using Point3f = Vector3f;
using Pointset3f = vector<pair<Point3f,Point3f>>;

class Image {
  
  private:
    int _rows;
    int _cols;
    cv::Mat _intensity;
    cv::Mat _depth;
    ArrayXf _xyz;

  public:
    Image() = default;
    explicit Image(int rows, int cols);

    inline const int rows() { return _rows; }
    inline const int cols() { return _cols; }
    inline cv::Mat& intensity() { return _intensity; }
    inline cv::Mat& depth() { return _depth; }
    inline const ArrayXf& xyz() { return _xyz; }
    inline float& xyz(int index) { return _xyz[index]; }
    inline const cv::Size size() { return _intensity.size(); }
    inline bool empty() { return _intensity.empty() || _depth.empty(); }
    inline Image clone() const {
      Image img;
      img._rows = _rows;
      img._cols = _cols;
      img._intensity = _intensity.clone();
      img._depth = _depth.clone();
      img._xyz = _xyz;
      return img;
    }
    inline Vector3f get3DPoint(cv::Point2f& point) {
      int index = (point.y *_cols + point.x)*3;
      return Vector3f(_xyz[index], _xyz[index+1], _xyz[index+2]);
    }
    void drawKeypoints(vector<cv::KeyPoint>& keypoints);
    void drawTracks(unordered_map<int,pair<vector<cv::Point2f>,cv::Scalar>>& tracks);
    static cv::Mat drawMatches(Image& img1, Image& img2, Pointset2f& matches, vector<bool>& inliers_mask);
};