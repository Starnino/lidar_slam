#pragma once

#include <sensor_msgs/PointCloud2.h>
#include <core/pointcloud.hpp>
#include <core/image.hpp>
#include <core/projector.hpp>
#include <opencv2/core.hpp>
#include <omp.h>

using Eigen::ArrayXi;

PointCloud deserializeCloudMsg(const sensor_msgs::PointCloud2::ConstPtr msg);
Image pointCloud2Img(PointCloud& cloud, Projector& projector);