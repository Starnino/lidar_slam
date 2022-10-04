add_executable(test_read_dataset test_read_dataset.cpp)
target_link_libraries(test_read_dataset ${catkin_LIBRARIES})

add_executable(test_lidar_projection test_lidar_projection.cpp)
target_link_libraries(test_lidar_projection core utils ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(test_extraction test_extraction.cpp)
target_link_libraries(test_extraction core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(test_tracking test_tracking.cpp)
target_link_libraries(test_tracking core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(test_ransac test_ransac.cpp)
target_link_libraries(test_ransac utils ${catkin_LIBRARIES})

add_executable(test_icp test_icp.cpp)
target_link_libraries(test_icp utils ${catkin_LIBRARIES})

add_executable(test_cloud_registration test_cloud_registration.cpp)
target_link_libraries(test_cloud_registration core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(test_lidar_trajectory test_lidar_trajectory.cpp)
set_source_files_properties(test_lidar_trajectory.cpp PROPERTIES COMPILE_FLAGS "-DWITHOUT_NUMPY" )
target_link_libraries(test_lidar_trajectory core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(test_gps_trajectory test_gps_trajectory.cpp)
set_source_files_properties(test_gps_trajectory.cpp PROPERTIES COMPILE_FLAGS "-DWITHOUT_NUMPY" )
target_link_libraries(test_gps_trajectory utils ${catkin_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(test_match_trajectory test_match_trajectory.cpp)
set_source_files_properties(test_match_trajectory.cpp PROPERTIES COMPILE_FLAGS "-DWITHOUT_NUMPY" )
target_link_libraries(test_match_trajectory core utils ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
