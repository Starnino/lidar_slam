add_executable(test_read_dataset test_read_dataset.cpp)
target_link_libraries(test_read_dataset ${catkin_LIBRARIES})

add_executable(test_lidar_projection test_lidar_projection.cpp)
target_link_libraries(test_lidar_projection core utils ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(test_superpoint_extraction test_superpoint_extraction.cpp)
target_link_libraries(test_superpoint_extraction core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(test_superpoint_tracking test_superpoint_tracking.cpp)
target_link_libraries(test_superpoint_tracking core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(test_ransac test_ransac.cpp)
target_link_libraries(test_ransac core utils ${catkin_LIBRARIES})

add_executable(test_cloud_registration test_cloud_registration.cpp)
target_link_libraries(test_cloud_registration core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
