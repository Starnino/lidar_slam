#include <iostream>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <utils/define.hpp>

using std::cout;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: read_dataset [PATH TO BAG FILE]\n";
    return 1;
  }

  int count = 0;

  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    
    if (m.getTopic() != CLOUD_TOPIC) continue;
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();  
    count++;
  }

  cout << "The bag file contains " << count << " messages of the topic " << CLOUD_TOPIC << "\n";

  bag.close();
  return 0;
} 