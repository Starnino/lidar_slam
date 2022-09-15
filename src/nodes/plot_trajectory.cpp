#include <iostream>
#include <fstream>
#include <string>
#include <utils/matplotlibcpp.hpp>
#include <utils/input_parser.hpp>
#include <core/registrator.hpp>
#include <utils/define.hpp>
#include <ros/package.h>

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
    cout << "Usage: lidar_projection [PATH TO TRAJECTORY FILE] [OPTIONS]\n";
    cout << "   -match  compares ransac and icp methods\n";
    cout << "   -save [NAME]  save plot with specified <name>\n";
    return 1;
  }
  string path = ros::package::getPath(PACKAGE_NAME);
  InputParser input(argc, argv);

  if (input.cmdOptionExists("-match")) {
    string filename(input.getArg(0));
    int pos = input.getArg(0).find("trajectory");
    string name = filename.substr(pos, 17);
    
    Pointset3f pointset;
    vector<float> x_gps, y_gps, z_gps;
    vector<float> x_icp_lidar, y_icp_lidar;
    std::ifstream icp_file(path + "/plots/icp/" + name);
    string line;
    while(std::getline(icp_file,line)) {
      if (line.size() > 0) {
        vector<float> nums = split<float>(line,',');
        Point3f gps_point(nums[0],nums[1],nums[2]);
        Point3f lidar_point(nums[3],nums[4],nums[5]);
        x_gps.push_back(nums[0]);
        y_gps.push_back(nums[1]);
        z_gps.push_back(nums[2]);
        x_icp_lidar.push_back(nums[3]);
        y_icp_lidar.push_back(nums[4]);
        pointset.push_back({gps_point, lidar_point});
      }
    }
    icp_file.close();
    
    vector<float> x_ransac_lidar, y_ransac_lidar;
    std::ifstream ransac_file(path + "/plots/ransac/" + name);
    while(std::getline(ransac_file,line)) {
      if (line.size() > 0) {
        vector<float> nums = split<float>(line,',');
        Point3f gps_point(nums[0],nums[1],nums[2]);
        Point3f lidar_point(nums[3],nums[4],nums[5]);
        x_ransac_lidar.push_back(nums[3]);
        y_ransac_lidar.push_back(nums[4]);
        pointset.push_back({gps_point, lidar_point});
      }
    }
    ransac_file.close();

    Affine3f pose = computeAlignment(pointset);
    for (size_t i = 0; i < x_gps.size(); ++i) {
      Point3f gps_point(x_gps[i],y_gps[i],z_gps[i]);
      Point3f gps_point_inlidarframe = pose*gps_point;
      x_gps[i] = gps_point_inlidarframe[0];
      y_gps[i] = gps_point_inlidarframe[1];
    }

    // plot trajectories
    plt::plot(x_gps, y_gps, {{"label", "gps"}});
    plt::plot(x_icp_lidar, y_icp_lidar, "tab:red", {{"label", "lidar-icp"}});
    plt::plot(x_ransac_lidar, y_ransac_lidar, "tab:green", {{"label", "lidar-ransac"}});
    plt::legend();
    plt::title("Trajectory");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");
    if (input.cmdOptionExists("-save")) {
      int pos = input.getArg(0).find("trajectory");
      string name = filename.substr(pos, 13);
      plt::savefig(path + "/plots/match-" + name + ".pdf");
    }
    plt::show();
  }

  else {
    
    string filename(input.getArg(0));
    bool icp = false;
    if (input.getArg(0).find("icp") != string::npos) icp = true; 
    
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

    Affine3f pose = computeAlignment(pointset);
    for (auto& pair : pointset) {
      Point3f gps_point_inlidarframe = pose*pair.first;
      Point3f lidar_point = pair.second;
      x_lidar.push_back(lidar_point[0]);
      y_lidar.push_back(lidar_point[1]);
      x_gps.push_back(gps_point_inlidarframe[0]);
      y_gps.push_back(gps_point_inlidarframe[1]);
    }

    // rmse error
    float l2 = 0.f;
    size_t size = x_lidar.size();
    for (size_t i = 0; i < size; ++i) {
      l2 += sqrt(pow(x_gps[i]-x_lidar[i],2) + pow(y_gps[i]-y_lidar[i],2));
    }
    float rmse = sqrt(l2/size);
    cout << "Root Mean Square Error = " << rmse << "\n";

    // plot trajectories
    plt::plot(x_gps, y_gps, {{"label", "gps"}});
    if (icp) plt::plot(x_lidar, y_lidar, "tab:red", {{"label", "lidar-icp"}});
    else plt::plot(x_lidar, y_lidar, "tab:green", {{"label", "lidar-ransac"}});
    plt::legend();
    plt::title("Trajectory");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");
    if (input.cmdOptionExists("-save")) {
      int pos = input.getArg(0).find("trajectory");
      string name = filename.substr(pos, 13);
      if (icp) plt::savefig(path + "/plots/icp-" + name + ".pdf");
      else plt::savefig(path + "/plots/ransac-" + name + ".pdf");
    }
    plt::show();
  }

  return 0;
}