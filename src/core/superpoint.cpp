/*
CREDITS:
Luca Di Gianmarino
*/
#include "superpoint.hpp"

#include <omp.h>

#include <Eigen/Dense>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <tuple>

using Eigen::ArrayXXi;
using std::vector;
using torch::max_pool2d;
using torch::relu;
using torch::softmax;
using torch::Tensor;
using torch::nn::Conv2d;
using torch::nn::Conv2dOptions;

struct SuperPointDetector::SuperPoint : torch::nn::Module {
  SuperPoint()
      : conv1a(Conv2dOptions(1, 64, 3).stride(1).padding(1)),
        conv1b(Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv2a(Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv2b(Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        conv3a(Conv2dOptions(64, 128, 3).stride(1).padding(1)),
        conv3b(Conv2dOptions(128, 128, 3).stride(1).padding(1)),
        conv4a(Conv2dOptions(128, 256, 3).stride(1).padding(1)),
        conv4b(Conv2dOptions(256, 256, 3).stride(1).padding(1)),
        convPa(Conv2dOptions(256, 256, 3).stride(1).padding(1)),
        convPb(Conv2dOptions(256, 65, 1).stride(1).padding(0)),
        convDa(Conv2dOptions(256, 256, 3).stride(1).padding(1)),
        convDb(Conv2dOptions(256, 256, 1).stride(1).padding(0)) {
    register_module("conv1a", conv1a);
    register_module("conv1b", conv1b);
    register_module("conv2a", conv2a);
    register_module("conv2b", conv2b);
    register_module("conv3a", conv3a);
    register_module("conv3b", conv3b);
    register_module("conv4a", conv4a);
    register_module("conv4b", conv4b);
    register_module("convPa", convPa);
    register_module("convPb", convPb);
    register_module("convDa", convDa);
    register_module("convDb", convDb);
  }

  std::tuple<Tensor, Tensor> Forward(Tensor x) {
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    auto cPa = relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]

    auto cDa = relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    // dense = np.exp(semi) # Softmax.
    // dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.

    semi = softmax(semi, 1);
    // semi = semi/(semi.sum()+0.000001);
    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]

    return std::make_tuple(semi, desc);
  }

  Tensor ForwardFeatures(Tensor x) {
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    auto cPa = relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]

    semi = softmax(semi, 1);

    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]

    return semi;
  }

  Tensor ForwardDescriptors(Tensor x) {
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    auto cDa = relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    return desc;
  }

  // ldg declaration of layers to be used in pred
  Conv2d conv1a;
  Conv2d conv1b;
  Conv2d conv2a;
  Conv2d conv2b;
  Conv2d conv3a;
  Conv2d conv3b;
  Conv2d conv4a;
  Conv2d conv4b;
  Conv2d convPa;
  Conv2d convPb;
  // ldg descriptors
  Conv2d convDa;
  Conv2d convDb;
};

SuperPointDetector::SuperPointDetector(const int& n_features,
                                       const float& threshold,
                                       const int& nms_dist,
                                       const bool& use_cuda,
                                       const std::string& sp_weights_file_name)
    : n_features_(n_features), threshold_(threshold), nms_dist_(nms_dist) {
  model_ = std::make_shared<SuperPoint>();

  if (sp_weights_file_name == "") {
    std::cout << "SuperPointDetector| could not find weight file\n";
    exit(1);
    
  } else {
    std::cout << "SuperPointDetector| Loading weight from: ";
    std::cout << sp_weights_file_name;
    std::cout << "\n";

    torch::load(model_, sp_weights_file_name);
  }

  bool use_cuda_ = use_cuda && torch::cuda::is_available();
  if (use_cuda_) {
    device_type_ = torch::kCUDA;
    std::cout << "SuperPointDetector| found CUDA! Go GPU\n";
  } else {
    device_type_ = torch::kCPU;
    std::cout << "SuperPointDetector| using CPU\n";
  }
};  

const float& SuperPointDetector::GetThreshold() const { return threshold_; };
void SuperPointDetector::SetThreshold(const float& threshold) {
  threshold_ = threshold;
};

const int& SuperPointDetector::GetMaxFeatures() const { return n_features_; };
void SuperPointDetector::SetMaxFeatures(const int& n_features) {
  n_features_ = n_features;
};

const int& SuperPointDetector::GetNMSDistance() const { return nms_dist_; };
void SuperPointDetector::SetNMSDistance(const int& nms_dist) {
  nms_dist_ = nms_dist;
};

void SuperPointDetector::EnableCuda() {
  bool use_cuda_ = torch::cuda::is_available();
  if (use_cuda_) {
    device_type_ = torch::kCUDA;
    std::cout << "SuperPointDetector| found CUDA! Go GPU\n";
  } else {
    device_type_ = torch::kCPU;
    std::cout << "SuperPointDetector| CUDA not found! Using CPU\n";
  }
};
void SuperPointDetector::DisableCuda() {
  device_type_ = torch::kCPU;
  std::cout << "SuperPointDetector| Using CPU\n";
};

int SuperPointDetector::FilterInvalidFeatures(
    const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) const {
  int position = 0;
  int features_removed = 0;

  for (const auto& keypoint : keypoints) {
    if (((keypoint.pt.x - feature_dim_ / 2) < 0) ||
        ((keypoint.pt.x + feature_dim_ / 2) > image.cols) ||
        ((keypoint.pt.y - feature_dim_ / 2) < 0) ||
        ((keypoint.pt.y + feature_dim_ / 2) > image.rows)) {
      keypoints.erase(keypoints.begin() + position);
      features_removed++;
      continue;
    }

    cv::Mat patch = image(cv::Rect(keypoint.pt.x - feature_dim_ / 2,
                                   keypoint.pt.y - feature_dim_ / 2,
                                   feature_dim_, feature_dim_));

    double min;
    cv::minMaxLoc(patch.reshape(1), &min);

    if (min < 0) {
      keypoints.erase(keypoints.begin() + position);
      features_removed++;
    } else {
      position++;
    }
  }
  return features_removed;
};

vector<cv::KeyPoint> SuperPointDetector::NonMaxSuppression(
    vector<cv::KeyPoint>& keypoints, const int& H, const int& W) const {
  if (keypoints.size() < 2) {
    return keypoints;
  }

  ArrayXXi grid =
      ArrayXXi::Zero(H + 2 * nms_dist_, W + 2 * nms_dist_);  // Track NMS data
  ArrayXXi response(H + 2 * nms_dist_, W + 2 * nms_dist_);
  // Sort by confidence decreasing mode
  std::sort(
      keypoints.begin(), keypoints.end(),
      [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response; });

// Initialize the grid.
#pragma omp parallel for
  for (const auto& keypoint : keypoints) {
    grid(keypoint.pt.y + nms_dist_, keypoint.pt.x + nms_dist_) = 1;
    response(keypoint.pt.y + nms_dist_, keypoint.pt.x + nms_dist_) =
        keypoint.response;
  }

  // Iterate through points, highest to lowest conf, suppress neighborhood.
  int x, y;
  for (const auto& keypoint : keypoints) {
    x = keypoint.pt.x + nms_dist_;
    y = keypoint.pt.y + nms_dist_;
    if (grid(y, x) == 1) {
      grid.block(y - nms_dist_, x - nms_dist_, 2 * nms_dist_ + 1,
                 2 * nms_dist_ + 1) = 0;
      grid(y, x) = -1;
    }
  }

  vector<cv::KeyPoint> result;
  // Get all surviving -1's
  for (int row = 0; row < H; ++row) {
    for (int col = 0; col < W; ++col) {
      if (grid(row, col) == -1)
        result.push_back(cv::KeyPoint(col - nms_dist_, row - nms_dist_,
                                      feature_dim_, -1, response(row, col)));
    }
  }

  return result;
};

void SuperPointDetector::detectAndCompute(const cv::Mat& image,
                                          vector<cv::KeyPoint>& keypoints,
                                          cv::Mat& descriptors) const {
  assert(image.depth() == CV_32F);

  Tensor img_tensor =
      torch::from_blob(image.clone().data, {1, 1, image.rows, image.cols},
                       torch::kFloat32)
          .clone();

  torch::Device device(device_type_);
  model_->to(device);
  img_tensor = img_tensor.set_requires_grad(false);

  // ldg prediction
  auto [m_prob, m_desc] =
      model_->Forward(img_tensor.to(device));  // ([H, W],[1, 256, H/8, W/8])
  auto prob = m_prob.squeeze(0);

  // keypoint
  keypoints.clear();
  auto kpts = (prob >= threshold_);
  // std::cerr << "sum of probabilities: " << prob.sum() << std::endl;
  kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)

  int num_keypoints = kpts.size(0);
  keypoints.resize(num_keypoints);
  for (int i = 0; i < num_keypoints; ++i) {
    const float& response = prob[kpts[i][0]][kpts[i][1]].item<float>();
    keypoints[i] =
        cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(),
                     feature_dim_, -1, response);
  }

  FilterInvalidFeatures(image, keypoints);

  if (nms_dist_ > 0)
    keypoints = NonMaxSuppression(keypoints, image.rows, image.cols);

  num_keypoints = keypoints.size();

  if (!(n_features_ < 0) && !(n_features_ > num_keypoints)) {
    std::sort(
        keypoints.begin(), keypoints.end(),
        [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response; });
    keypoints.resize(n_features_);
    num_keypoints = n_features_;
  }

  // descriptors
  cv::Mat kpt_mat(num_keypoints, 2, CV_32F);  // [n_keypoints, 2]  (y, x)

  for (int i = 0; i < num_keypoints; ++i) {
    kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
    kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
  }

  auto fkpts =
      torch::from_blob(kpt_mat.data, {num_keypoints, 2}, torch::kFloat);

  auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
  grid = grid.to(device_type_);

  grid[0][0].slice(1, 0, 1) =
      2.0f * fkpts.slice(1, 1, 2) / image.cols - 1;  // x
  grid[0][0].slice(1, 1, 2) =
      2.0f * fkpts.slice(1, 0, 1) / image.rows - 1;  // y

  auto desc = torch::grid_sampler(m_desc, grid, 0, 0,
                                  false);  // [1, 256, 1, n_keypoints]
  desc = desc.squeeze(0).squeeze(1);       // [256, n_keypoints]

  // normalize to 1
  auto dn = torch::norm(desc, 2, 1);
  desc = desc.div(torch::unsqueeze(dn, 1));

  desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
  desc = desc.to(torch::kCPU);
  cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1,
                   desc.data_ptr<float>());

  descriptors = desc_mat.clone();
};

void SuperPointDetector::detect(const cv::Mat& image,
                                vector<cv::KeyPoint>& keypoints) const {
  assert(image.depth() == CV_32F);

  Tensor img_tensor =
      torch::from_blob(image.clone().data, {1, 1, image.rows, image.cols},
                       torch::kFloat32)
          .clone();

  torch::Device device(device_type_);
  model_->to(device);
  img_tensor = img_tensor.set_requires_grad(false);

  // ldg prediction
  auto m_prob = model_->ForwardFeatures(
      img_tensor.to(device));  // ([H, W],[1, 256, H/8, W/8])
  auto prob = m_prob.squeeze(0);

  // keypoint
  keypoints.clear();
  auto kpts = (prob >= threshold_);
  // std::cerr << "sum of probabilities: " << prob.sum() << std::endl;
  kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)

  int num_keypoints = kpts.size(0);
  keypoints.resize(num_keypoints);
  for (int i = 0; i < num_keypoints; ++i) {
    const float& response = prob[kpts[i][0]][kpts[i][1]].item<float>();
    keypoints[i] =
        cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(),
                     feature_dim_, -1, response);
  }

  FilterInvalidFeatures(image, keypoints);

  if (nms_dist_ > 0)
    keypoints = NonMaxSuppression(keypoints, image.rows, image.cols);

  num_keypoints = keypoints.size();

  if (!(n_features_ < 0) && !(n_features_ > num_keypoints)) {
    std::sort(
        keypoints.begin(), keypoints.end(),
        [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response; });
    keypoints.resize(n_features_);
    num_keypoints = n_features_;
  }
};

void SuperPointDetector::compute(const cv::Mat& image,
                                 vector<cv::KeyPoint>& keypoints,
                                 cv::Mat& descriptors) const {
  assert(image.depth() == CV_32F);

  Tensor img_tensor =
      torch::from_blob(image.clone().data, {1, 1, image.rows, image.cols},
                       torch::kFloat32)
          .clone();

  torch::Device device(device_type_);
  model_->to(device);
  img_tensor = img_tensor.set_requires_grad(false);

  // ldg prediction
  auto m_desc = model_->ForwardDescriptors(
      img_tensor.to(device));  // ([H, W],[1, 256, H/8, W/8])

  int num_keypoints = static_cast<int>(keypoints.size());
  cv::Mat kpt_mat(num_keypoints, 2, CV_32F);  // [n_keypoints, 2]  (y, x)

  for (int i = 0; i < num_keypoints; ++i) {
    kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
    kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
  }

  auto fkpts =
      torch::from_blob(kpt_mat.data, {num_keypoints, 2}, torch::kFloat);

  auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
  grid = grid.to(device_type_);

  grid[0][0].slice(1, 0, 1) =
      2.0f * fkpts.slice(1, 1, 2) / image.cols - 1;  // x
  grid[0][0].slice(1, 1, 2) =
      2.0f * fkpts.slice(1, 0, 1) / image.rows - 1;  // y

  auto desc = torch::grid_sampler(m_desc, grid, 0, 0,
                                  false);  // [1, 256, 1, n_keypoints]
  desc = desc.squeeze(0).squeeze(1);       // [256, n_keypoints]

  // normalize to 1
  auto dn = torch::norm(desc, 2, 1);
  desc = desc.div(torch::unsqueeze(dn, 1));

  desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
  desc = desc.to(torch::kCPU);
  cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1,
                   desc.data_ptr<float>());

  descriptors = desc_mat.clone();
};
