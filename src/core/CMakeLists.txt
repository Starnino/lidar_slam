add_library(core 
  pointcloud.cpp pointcloud.hpp
  cvimage.cpp cvimage.hpp
  projector.cpp projector.hpp
  superpoint.cpp superpoint.hpp
  tracker.cpp tracker.hpp)

target_link_libraries(core 
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  OpenMP::OpenMP_CXX)
