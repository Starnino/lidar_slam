#include "tracker.hpp"

Tracker::Tracker(Matcher matcher, float knn_threshold, int norm_threshold, int norm_type) {
  switch (matcher) {
  
  case Matcher::BFMatcher:
    _matcher = cv::BFMatcher::create(norm_type);
    break;
  
  case Matcher::FLANNMatcher:
    _matcher = cv::FlannBasedMatcher::create(); 
    break;
  }

  _knn_threshold = knn_threshold;
  _norm_threshold = norm_threshold;
}

vector<cv::DMatch> Tracker::knnmatch(cv::Mat& descriptors1, cv::Mat& descriptors2) {
  assert(descriptors1.cols == descriptors2.cols);
  
  vector<vector<cv::DMatch>> knn_matches;
  vector<cv::DMatch> good_matches;

  _matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

  for (size_t i = 0; i < knn_matches.size(); ++i) {
    if (knn_matches[i][0].distance < _knn_threshold * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
 
  return good_matches;
}

tuple<Pointset2f,Pointset3f> Tracker::update(vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, Image& img) {
  // remove invalid points
  vector<Point3f> new_points; vector<cv::KeyPoint> new_keypoints; cv::Mat new_descriptors;
  for (size_t i = 0; i < keypoints.size(); ++i) {
    Point3f point = img.get3DPoint(keypoints[i].pt);
    if (point.isZero()) continue;
    new_points.push_back(point);
    new_keypoints.push_back(keypoints[i]);
    new_descriptors.push_back(descriptors.row(i));
  }
  
  if (_last_points.size() < 3) {
    _last_points = new_points;
    _last_keypoints = new_keypoints;
    _last_descriptors = new_descriptors;
    return {};
  }

  if (new_points.size() < 3) {
    return {};
  }

  vector<cv::DMatch> matches = knnmatch(_last_descriptors, new_descriptors);
  
  Pointset2f correspondeces2D;
  Pointset3f correspondeces3D;
  for (size_t i = 0; i < matches.size(); ++i) {
    int query_idx = matches[i].queryIdx;
    int train_idx = matches[i].trainIdx;
    cv::Point2f query_point = _last_keypoints[query_idx].pt;
    cv::Point2f train_point = new_keypoints[train_idx].pt;
    if (cv::norm(query_point-train_point) < _norm_threshold) {
      correspondeces2D.push_back({query_point, train_point});
      correspondeces3D.push_back({_last_points[query_idx], new_points[train_idx]});
    }
  }

  _last_points = new_points;
  _last_keypoints = new_keypoints;
  _last_descriptors = new_descriptors;
  
  return {correspondeces2D, correspondeces3D}; 
}

unordered_map<int,pair<vector<cv::Point2f>,cv::Scalar>> Tracker::updateTracks(vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  // inizialize tracks
  if (_tracks.empty()) {
    for (size_t k = 0; k < keypoints.size(); ++k) {
      cv::Scalar color(
        (float) std::rand() / RAND_MAX,
        (float) std::rand() / RAND_MAX,
        (float) std::rand() / RAND_MAX
      );
      _tracks[k] = {{keypoints[k].pt}, color};
    }
    _last_descriptors = descriptors.clone(); 
    return _tracks;
  }
  
  // update tracks
  vector<cv::DMatch> matches = knnmatch(_last_descriptors, descriptors);
  cv::Mat new_descriptors(_last_descriptors.size(), _last_descriptors.type());
  vector<cv::Mat> unmatched_descriptors;
  vector<int> last_mask(_last_descriptors.size().height, 1);
  vector<int> new_mask(descriptors.size().height, 1);
  set<int> usedIdx;

  for (size_t i = 0; i < matches.size(); ++i) {
    int queryIdx = matches[i].queryIdx;
    int trainIdx = matches[i].trainIdx;
    _tracks[queryIdx].first.push_back(keypoints[trainIdx].pt);
    descriptors.row(trainIdx).copyTo(new_descriptors.row(queryIdx));
    last_mask[queryIdx] = 0; new_mask[trainIdx] = 0;
    usedIdx.insert(queryIdx);
  }
  
  for (size_t i = 0; i < new_mask.size(); ++i) {
    if (new_mask[i]) {
      unmatched_descriptors.push_back(descriptors.row(i));
    } 
  }

  for (size_t i = 0; i < last_mask.size(); ++i) {
    if (last_mask[i]) {
      if (unmatched_descriptors.size() == 0) break;
      unmatched_descriptors.back().copyTo(new_descriptors.row(i));
      unmatched_descriptors.pop_back();
    }
  }
  
  // eliminate ending tracks
  for (auto& track : _tracks) {
    if (!usedIdx.count(track.first)) {
      track.second.first.clear();
    }
  }
  
  _last_descriptors = new_descriptors.clone();

  return _tracks;
}