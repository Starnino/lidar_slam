add_library(utils
    input_parser.cpp input_parser.hpp
    cloud_helper.cpp cloud_helper.hpp
    json_helper.cpp json_helper.hpp
    gps_helper.cpp gps_helper.hpp
    ransac.cpp ransac.hpp 
    define.hpp
    icp.cpp icp.hpp
    matplotlibcpp.hpp)
    
target_link_libraries(utils 
    core 
    OpenMP::OpenMP_CXX
    ${JSONCPP_LIBRARIES})

add_executable(plot_trajectory plot_trajectory)
set_source_files_properties(plot_trajectory PROPERTIES COMPILE_FLAGS "-DWITHOUT_NUMPY" )
target_link_libraries(plot_trajectory utils ${catkin_LIBRARIES} ${PYTHON_LIBRARIES})