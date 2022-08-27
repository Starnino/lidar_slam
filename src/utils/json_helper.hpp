#pragma once

#include <fstream>
#include <jsoncpp/json/json.h>
#include <string>
#include <tuple>
#include <core/projector.hpp>
#include <core/superpoint.hpp>

using std::string;
using std::tuple;

namespace json {
  Projector loadProjectorConfig(string path);
  SuperPointDetector loadSuperPointConfig(string path, string config_filename);
  tuple<int,float> loadRANSACConfig(string path);
}