cmake_minimum_required(VERSION 3.0.2)
project(lidarslam)

# c++ standard
add_compile_options(-std=c++17)
# level of optimization 
add_compile_options(-O3) 
# enable cpu parallelism
add_compile_options(-msse2)
# no warnings
add_compile_options(-Wall)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs rosbag)
message("${PROJECT_NAME}|ROS components found in ${catkin_INCLUDE_DIRS}")

## Find Eigen
find_package(Eigen3 REQUIRED)
message("${PROJECT_NAME}|Eigen3 version ${Eigen3_VERSION} found in ${Eigen3_DIR}")

## Find OpenCV
find_package(OpenCV REQUIRED)
message("${PROJECT_NAME}|using OpenCV version ${OpenCV_VERSION} found in (${OpenCV_DIR})")

## Find JSONCPP
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
message("${PROJECT_NAME}|using JSONCPP version ${JSONCPP_VERSION}")

# Find OMP
find_package(OpenMP REQUIRED)
message("${PROJECT_NAME}|using OpenMP found = ${OpenMP_CXX_FOUND}")

# Find Torch without a torch
set(Torch_DIR $ENV{TORCH_DIR})
find_package(Torch REQUIRED)
message("${PROJECT_NAME}|Torch headers in (${TORCH_INCLUDE_DIRS})")
message("${PROJECT_NAME}|Torch libraries in (${TORCH_LIBRARIES})")
add_compile_options(${TORCH_CXX_FLAGS})

## catkin specific configuration
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS src
   LIBRARIES core utils
   CATKIN_DEPENDS roscpp sensor_msgs rosbag
   DEPENDS )

## Search include files
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src)   

## Link CMakeLists files recursively
add_subdirectory(${PROJECT_SOURCE_DIR}/src)