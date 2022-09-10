#pragma once

#include <unordered_map>
#include <set>
#include <tuple>
#include <core/image.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

using std::vector;
using std::unordered_map;
using std::set;
using std::pair;
using std::tuple;
using Pointset2f = vector<pair<cv::Point2f,cv::Point2f>>;
using Descriptor = Eigen::VectorXf;

enum class Matcher {BFMatcher, FLANNMatcher};

class Tracker {

  private:
    cv::Ptr<cv::DescriptorMatcher> _matcher;
    float _knn_threshold;
    int _norm_threshold;
    cv::Mat _last_descriptors;
    vector<cv::KeyPoint> _last_keypoints;
    vector<Point3f> _last_points;
    unordered_map<int, pair<vector<cv::Point2f>,cv::Scalar>> _tracks;
    vector<cv::DMatch> match(cv::Mat& descriptors1, cv::Mat& descriptors2);

  public:
    explicit Tracker(Matcher matcher, float knn_threshold, int norm_threshold);

    tuple<Pointset2f,Pointset3f> update(vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, Image& img);
    unordered_map<int,pair<vector<cv::Point2f>,cv::Scalar>> updateTracks(vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void drawTracks();
};