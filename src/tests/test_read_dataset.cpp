#include <iostream>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

#define TOPIC "/os1_cloud_node/points"

using std::cout;

int main(int argc, char **argv) {
  
  if (argc < 2) {
    cout << "Usage: read_dataset [PATH TO BAG FILE]\n";
    return 1;
  }

  int count = 0;

  rosbag::Bag bag(argv[1]);
  for (rosbag::MessageInstance const m: rosbag::View(bag)) {
    
    if (m.getTopic() != TOPIC) continue;
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();  
    count++;
  }

  cout << "The bag file contains " << count << " messages of the topic " << TOPIC << "\n";

  bag.close();
  return 0;
} 