#include "image.hpp"

Image::Image(int rows, int cols) {
    _rows = rows;
    _cols = cols;
    _intensity = cv::Mat::ones(rows, cols, CV_32F) * INV_CV_32F;
    _depth = cv::Mat::ones(rows, cols, CV_32F) * INV_CV_32F;
    _xyz = ArrayXf::Zero(rows*cols*3);
}

void Image::drawKeypoints(std::vector<cv::KeyPoint>& keypoints) {
    for (const cv::KeyPoint &keypoint : keypoints) {
      cv::circle(_intensity, keypoint.pt, 2, cv::Scalar(0.f, 0.f, 1.f), cv::FILLED);
      cv::circle(_depth, keypoint.pt, 2, cv::Scalar(0.f, 0.f, 1.f), cv::FILLED);
    }
}

void Image::drawTracks(unordered_map<int,pair<vector<cv::Point2f>,cv::Scalar>>& tracks) {
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

cv::Mat Image::drawMatches(Image& img1, Image& img2, Pointset2f& matches, vector<bool>& inliers_mask) {
  cv::Size size(img2.size().width, img2.size().height*2);
  cv::Mat color_img = cv::Mat::zeros(size, CV_32F);

  if (!(img1.empty() || img2.empty() || inliers_mask.size() == 0)) {
    cv::Mat img = cv::Mat::zeros(size, CV_32F);
    cv::vconcat(img1.intensity(), img2.intensity(), img);
    cv::cvtColor(img, color_img, cv::COLOR_GRAY2RGB);

    int i = 0;
    for (pair<cv::Point2f,cv::Point2f>& pointpair : matches) {
      cv::Point point1 = pointpair.first;
      cv::Point point2(pointpair.second.x, pointpair.second.y + img2.size().height);
      cv::Scalar color = inliers_mask[i] ? cv::Scalar(0.f, 1.f, 0.f) : cv::Scalar(0.f, 0.f, 1.f);
      cv::circle(color_img, point1, 2, color, cv::FILLED);
      cv::circle(color_img, point2, 2, color, cv::FILLED);
      if (inliers_mask[i]) cv::line(color_img, point1, point2, color, 1);
      i++;
    } 
  }
  return color_img;
}