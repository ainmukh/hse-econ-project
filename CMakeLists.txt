cmake_minimum_required(VERSION 3.16)
project(cw)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER "clang++")

add_executable(project logistic_regression.cpp
        Approximation_prev.h
        approx_logistic_regression.cpp
        SGD.h
        Approximation.h
        test.cpp
        Approximation_alex.h main.cpp)
include_directories("/usr/local/include/eigen3")
include_directories("/usr/local/LBFGSpp-master/include")
include_directories("/usr/local/include")
include_directories("/usr/local/include/gsl")
include_directories("/usr/local/lib/python3.9/site-packages/numpy/core/include")
target_link_libraries(project -L/usr/local/lib/ gsl)
target_link_libraries(project -L/usr/local/lib/ gslcblas)
#target_link_libraries(project -L/usr/local/lib/python3.9/site-packages/ numpy)

find_package (Python COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
target_link_libraries(project ${Python_LIBRARIES})
