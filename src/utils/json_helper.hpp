#pragma once

#include <fstream>
#include <jsoncpp/json/json.h>
#include <string>
#include <tuple>
#include <core/projector.hpp>
#include <core/superpoint.hpp>
#include <core/tracker.hpp>
#include <opencv2/features2d.hpp>

using std::string;
using std::tuple;

namespace json {
  Projector loadProjectorConfig(string path);
  SuperPointDetector loadSuperPointConfig(string path, string config_filename);
  cv::Ptr<cv::FeatureDetector> loadORBConfig(string path);
  tuple<int,float> loadRANSACConfig(string path);
  Tracker loadMatchConfig(string path);
}