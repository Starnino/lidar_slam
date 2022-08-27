/*
CREDITS:
Luca Di Gianmarino
*/
#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

class SuperPointDetector {
 public:
  explicit SuperPointDetector(const int& n_features = 500,
                              const float& threshold = 0.2f,
                              const int& nms_dist = 4,
                              const bool& use_cuda = false,
                              const std::string& sp_weights_file_name = "");
  const float& GetThreshold() const;
  void SetThreshold(const float& threshold);
  const int& GetNMSDistance() const;
  void SetNMSDistance(const int& nms_dist);
  const int& GetMaxFeatures() const;
  void SetMaxFeatures(const int& n_features);
  void EnableCuda();
  void DisableCuda();
  void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) const;
  void compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
               cv::Mat& descriptors) const;
  void detectAndCompute(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors) const;

 private:
  struct SuperPoint;
  std::vector<cv::KeyPoint> NonMaxSuppression(
      std::vector<cv::KeyPoint>& keypoints, const int& H, const int& W) const;
  int FilterInvalidFeatures(const cv::Mat& image,
                            std::vector<cv::KeyPoint>& keypoints) const;
  std::shared_ptr<SuperPoint> model_;
  int n_features_;
  float threshold_;
  int nms_dist_;
  int feature_dim_ = 3;
  torch::DeviceType device_type_ = torch::kCPU;  // default is on CPU
};
