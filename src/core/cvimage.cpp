#include "cvimage.hpp"

CVImage::CVImage(int rows, int cols) {
    _rows = rows;
    _cols = cols;
    _intensity = cv::Mat::ones(rows, cols, CV_32F) * INV_CV_32F;
    _depth = cv::Mat::ones(rows, cols, CV_32F) * INV_CV_32F;
    _xyz = ArrayXf::Zero(rows*cols*3);
}

void CVImage::drawKeypoints(std::vector<cv::KeyPoint>& keypoints) {
    for (const cv::KeyPoint &keypoint : keypoints) {
      cv::circle(_intensity, keypoint.pt, 2, cv::Scalar(0.f, 0.f, 1.f), cv::FILLED);
      cv::circle(_depth, keypoint.pt, 2, cv::Scalar(0.f, 0.f, 1.f), cv::FILLED);
    }
}

void CVImage::drawTracks(unordered_map<int,pair<vector<cv::Point2f>,cv::Scalar>> tracks) {
  for (auto elem : tracks) {
    vector<cv::Point2f> points = elem.second.first;
    cv::Scalar color = elem.second.second;
    if (points.size() > 1) {
      for (size_t i = 0; i < points.size(); ++i) {
        cv::circle(_intensity, points[i], 2, color, cv::FILLED);
        cv::circle(_depth, points[i], 2, color, cv::FILLED);
      } 
    }
  } 
}