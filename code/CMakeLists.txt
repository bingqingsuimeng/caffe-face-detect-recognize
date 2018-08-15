cmake_minimum_required(VERSION 2.8)
project(FACE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES_1 highway_test.cc FACE.cpp FACE.h insightface.h insightface.cpp)
add_executable(Highway ${SOURCE_FILES_1})

target_link_libraries(Highway ${OpenCV_LIBS} )
target_link_libraries(Highway ${Caffe_LIBRARIES})

#set(SOURCE_F input_pair.cpp FACE.cpp FACE.h)
#add_executable(Input_t ${SOURCE_F})

#target_link_libraries(Input_t ${OpenCV_LIBS} )
#target_link_libraries(Input_t ${Caffe_LIBRARIES})

#set(ID_False_S id2photo_test.cc FACE.cpp FACE.h)
#add_executable(ID_False ${ID_False_S})

#target_link_libraries(ID_False ${OpenCV_LIBS} )
#target_link_libraries(ID_False ${Caffe_LIBRARIES})

#set(Head_Estimate_S headpose_demo.cpp pose_estimate.cpp pose_estimate.h FACE.cpp FACE.h)
#add_executable(Head_Estimate ${Head_Estimate_S})

#target_link_libraries(Head_Estimate ${OpenCV_LIBS} )
#target_link_libraries(Head_Estimate ${Caffe_LIBRARIES})

set(SOURCE_FILES lbp_test.cc FACE.cpp FACE.h face_info.h remove_duplicate.h list.h remove_duplicate.cc )
add_executable(LBPH ${SOURCE_FILES})

target_link_libraries(LBPH ${OpenCV_LIBS} )
target_link_libraries(LBPH ${Caffe_LIBRARIES})
