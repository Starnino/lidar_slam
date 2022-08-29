add_library(utils 
    cloud_helper.cpp cloud_helper.hpp
    json_helper.cpp json_helper.hpp
    ransac.hpp)
    
target_link_libraries(utils 
    core 
    OpenMP::OpenMP_CXX
    ${JSONCPP_LIBRARIES})