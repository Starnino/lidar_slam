add_library(core 
  pointcloud.cpp pointcloud.hpp
  image.cpp image.hpp
  projector.cpp projector.hpp
  superpoint.cpp superpoint.hpp
  tracker.cpp tracker.hpp
  registrator.cpp registrator.hpp)

target_link_libraries(core 
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  OpenMP::OpenMP_CXX)
