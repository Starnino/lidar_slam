#pragma once

#include <unordered_map>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


using std::vector;
using std::unordered_map;
using std::set;
using std::pair;
using Pointset2f = vector<pair<cv::Point2f,cv::Point2f>>;

enum class Matcher {BFMatcher, FLANNMatcher};

class Tracker {

  private:
    cv::Ptr<cv::DescriptorMatcher> _matcher;
    cv::Mat _last_descriptors;
    vector<cv::KeyPoint> _last_keypoints;
    float _knn_threshold;
    unordered_map<int, pair<vector<cv::Point2f>,cv::Scalar>> _tracks;
    vector<cv::DMatch> match(cv::Mat& descriptors1, cv::Mat& descriptors2);

  public:
    explicit Tracker(Matcher matcher, float knn_threshold = 0.7f);

    Pointset2f update(vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    unordered_map<int,pair<vector<cv::Point2f>,cv::Scalar>> updateTracks(vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void drawTracks();
};