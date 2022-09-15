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