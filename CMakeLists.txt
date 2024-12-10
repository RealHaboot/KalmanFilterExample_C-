cmake_minimum_required(VERSION 3.30)
project(Kalman_Filter_V_1)

set(CMAKE_CXX_STANDARD 20)

# Add the Eigen include path
include_directories("C:/Users/sgndd/OneDrive/Documents/C++ Libraries/eigen-3.4.0/eigen-3.4.0") #Depending on where in your files the eigen library is you will need to change this

add_executable(Kalman_Filter_V_1 main.cpp)
