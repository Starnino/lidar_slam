#include <iostream>
#include <fstream>
#include <string>
#include <utils/matplotlibcpp.hpp>
#include <utils/input_parser.hpp>
#include <core/registrator.hpp>

using std::cout;
using std::string;
namespace plt = matplotlibcpp;

template <typename T>
vector<T> split(string str, char separator) {
  vector<T> v;
  int currIndex = 0, i = 0;  
  int startIndex = 0, endIndex = 0;
  int len = str.size();  
  while (i <= len) {  
    if (str[i] == separator || i == len) {  
      endIndex = i;  
      string subStr = "";  
      subStr.append(str, startIndex, endIndex - startIndex);  
      v.push_back(std::stof(subStr));  
      currIndex += 1;  
      startIndex = endIndex + 1;  
    }  
    i++;  
  } return v;   
}

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO TRAJECTORY FILE] [OPTION]\n";
    cout << "   -s   save file to the same path";
    return 1;
  }
  InputParser input(argc, argv);
  string filename(input.getArg(0));
  bool save = false;
  if (input.cmdOptionExists("-s")) save = true;
  
  std::ifstream file(filename);
  if(!file) {
    std::stringstream stream;
    stream << "failed to open file " << filename << '\n';
    throw std::runtime_error(stream.str());
  }

  Pointset3f pointset;
  string line;
  while(std::getline(file,line)) {
    if (line.size() > 0) {
      vector<float> nums = split<float>(line,',');
      Point3f gps_point(nums[0],nums[1],nums[2]);
      Point3f lidar_point(nums[3],nums[4],nums[5]);
      pointset.push_back({gps_point, lidar_point});
    }
  }
  file.close();
  
  vector<float> x_lidar, y_lidar;
  vector<float> x_gps, y_gps;

  Affine3f pose = Registrator::computeAlignment(pointset);
  for (auto& pair : pointset) {
    Point3f gps_point_inlidarframe = pose*pair.first;
    Point3f lidar_point = pair.second;
    x_lidar.push_back(lidar_point[0]);
    y_lidar.push_back(lidar_point[1]);
    x_gps.push_back(gps_point_inlidarframe[0]);
    y_gps.push_back(gps_point_inlidarframe[1]);
  }

  // plot trajectories
  plt::plot(x_gps, y_gps, {{"label", "gps"}});
  plt::plot(x_lidar, y_lidar, {{"label", "lidar"}});
  plt::legend();
  plt::title("Trajectory");
  plt::xlabel("x [m]");
  plt::ylabel("y [m]");
  if (save) plt::savefig(filename.substr(0,filename.size()-3) + "pdf");
  plt::show();

  return 0;
}