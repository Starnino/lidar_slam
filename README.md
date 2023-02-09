# Sparse LiDAR Odometry using intensity channel: a comparison
Project repository of the thesis project for Msc in Artificial Intelligence and Robotics, Sapienza University of Rome. 

## Abstract
In the latest years, computer vision is becoming one of the most important Artificial Intelligence field due to the variety of practical applications, especially the ones concerning robot navigation systems. Through the use of these sensors, it is possible to provide to a navigation system the capability of estimating the current position in real-time, which can be useful in many scenarios. This thesis project focuses on the use of LiDAR 3D data for estimating the odometry of a vehicle, giving attention to the different available methods in order to make a comparison.

## The project
This thesis project focuses on the exploitation of LiDAR information using a classical 2D approach; first the point clouds are projected into a 2D image, then feature extraction techniques are applied in order to extract a bunch of interesting points which can be detected in subsequent LiDAR frames. The points are extracted using a traditional approach, Oriented FAST and Rotated BRIEF (ORB), or a newly deep learning approach called SuperPoint. The process of finding the correspondences between interesting points of previous and next cloud is called tracking. Once the interesting points have been coupled, one can retrieve the 3D correspondences and perform registration techniques for finding the rigid body transformation between the previous and the next cloud, such as Random Sample Consensus (RANSAC) or Iterative Reweighed Least-Squares (IRLS). The transformations are then accumulated in order to estimate the trajectory travelled by the car. The experiments have been done on the IPB Car dataset.

## Pipeline
- Project 3D LiDAR point clouds to a 2D image
- 2D features detection and description
- Find associations by brute-force knn algorithm
- Retrieve 3D points and perform registration
- Accumulate rigid body transformations for estimating odometry

## Examples
[![estimation.png](https://i.postimg.cc/kXkhwYRJ/estimation.png)](https://postimg.cc/1frH3HK2)

## [Report](https://github.com/Starnino/lidar_slam/blob/main/Sparse_LiDAR_Odometry.pdf)
