add_executable(feature_extractor feature_extractor.cpp)
target_link_libraries(feature_extractor core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(lidar_odometry_publisher lidar_odometry_publisher.cpp)
target_link_libraries(lidar_odometry_publisher core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(gps_odometry_publisher gps_odometry_publisher.cpp)
target_link_libraries(gps_odometry_publisher utils ${catkin_LIBRARIES})

add_executable(odometry2file odometry2file.cpp)
target_link_libraries(odometry2file utils ${catkin_LIBRARIES})
