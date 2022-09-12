#include <iostream>
#include <utils/gps_helper.hpp>
#include <utils/define.hpp>
#include <utils/matplotlibcpp.hpp>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

using std::cout;
using std::vector;
namespace plt = matplotlibcpp;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: lidar_projection [PATH TO BAG FILE]\n";
    return 1;
  }

  vector<float> x_gps; vector<float> y_gps;
  double initial_x = 0.0;
  double initial_y = 0.0;
  bool first = true;
  
  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    if (m.getTopic() != GPS_TOPIC) continue;
    
    sensor_msgs::NavSatFix::ConstPtr gps_msg = m.instantiate<sensor_msgs::NavSatFix>();
    double latitude = gps_msg->latitude;
    double longitude = gps_msg->longitude;
    auto [x, y, z] = geodetic2ecef(latitude, longitude);

    if (first) {
      initial_x = x;
      initial_y = y;
      first = false;
    }

    x_gps.push_back(x-initial_x);
    y_gps.push_back(y-initial_y);
  }
  bag.close();

  // plot trajectories
  plt::plot(x_gps, y_gps);
  plt::title("GPS Trajectory");
  plt::xlabel("x [m]");
  plt::ylabel("y [m]"); 
  plt::show();

  return 0;
}