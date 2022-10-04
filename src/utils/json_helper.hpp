#pragma once

#include <fstream>
#include <jsoncpp/json/json.h>
#include <string>
#include <tuple>

using std::string;
using std::tuple;

namespace json {
  tuple<int,int,float,float,float,float> loadProjectorConfig(string path);
  tuple<int,float,float,string> loadSuperPointConfig(string path, string config_filename);
  tuple<int,float,int,int,int,int> loadORBConfig(string path);
  tuple<int,float> loadRANSACConfig(string path);
  tuple<int,float, float, float> loadICPConfig(string path);
  tuple<string,float,int,int> loadMatchConfig(string path);
}