#include "json_helper.hpp"

Projector json::loadProjectorConfig(string path) {
  Json::Reader reader;
  Json::Value cfg;
  std::ifstream file(path);
  reader.parse(file, cfg);

  int height = cfg["Os1Lidar"]["vertical_resolution"].asInt();
  int width = cfg["Os1Lidar"]["horizon_resolution"].asInt();
  float fov_up = cfg["Os1Lidar"]["fov"]["vertical_up"].asFloat();
  float fov_down = cfg["Os1Lidar"]["fov"]["vertical_down"].asFloat();
  float max_depth = cfg["Os1Lidar"]["max_depth"].asFloat();
  float max_intensity = cfg["Os1Lidar"]["max_intensity"].asFloat();

  return Projector(height, width, fov_up, fov_down, max_depth, max_intensity);
}

SuperPointDetector json::loadSuperPointConfig(string path, string config_filename) {
  Json::Reader reader;
  Json::Value cfg;
  std::ifstream file(path + config_filename);
  reader.parse(file, cfg);

  string weights_file = cfg["SuperPoint"]["weights_file"].asString();
  float threshold = cfg["SuperPoint"]["threshold"].asFloat();
  float nms_dist = cfg["SuperPoint"]["nms_dist"].asFloat();

  return SuperPointDetector(-1, threshold, nms_dist, false, path + weights_file);
}

cv::Ptr<cv::FeatureDetector> json::loadORBConfig(string path) {
  Json::Reader reader;
  Json::Value cfg;
  std::ifstream file(path);
  reader.parse(file, cfg);

  int nfeatures = cfg["ORB"]["nfeatures"].asInt();
  float scale = cfg["ORB"]["scale"].asFloat();
  int nlevels = cfg["ORB"]["nlevels"].asInt();
  int edge_threshold = cfg["ORB"]["edge_threshold"].asInt();
  int patch_size = cfg["ORB"]["patch_size"].asInt();
  int fast_threshold = cfg["ORB"]["fast_threshold"].asInt();

  return cv::ORB::create(nfeatures, scale, nlevels, edge_threshold, 0, 2, cv::ORB::HARRIS_SCORE, patch_size, fast_threshold);
}

tuple<int,float> json::loadRANSACConfig(string path) {
  Json::Reader reader;
  Json::Value cfg;
  std::ifstream file(path);
  reader.parse(file, cfg);

  int iterations = cfg["RANSAC"]["iterations"].asInt();
  float inliers_threshold = cfg["RANSAC"]["inliers_threshold"].asFloat();

  return {iterations, inliers_threshold};
}

Tracker json::loadMatchConfig(string path) {
  Json::Reader reader;
  Json::Value cfg;
  std::ifstream file(path);
  reader.parse(file, cfg);

  string type = cfg["Matcher"]["type"].asString();
  Matcher matcher = type == "brute-force" ? Matcher::BFMatcher : Matcher::FLANNMatcher;
  float knn_threshold = cfg["Matcher"]["knn_threshold"].asFloat();
  int norm_threshold = cfg["Matcher"]["norm_threshold"].asInt();
  int norm_type = cfg["Matcher"]["norm_type"].asInt();

  return Tracker(matcher, knn_threshold, norm_threshold, norm_type);
}