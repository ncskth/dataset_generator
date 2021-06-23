cmake_minimum_required(VERSION 3.0.2)
project(gazebo_modelstates_timestamp_patch)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  gazebo_ros
  roscpp
  std_msgs
  message_generation
)
find_package(gazebo REQUIRED)

add_message_files(
	FILES
	  model_states_with_timestamp.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  gazebo_msgs
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS message_runtime gazebo_msgs gazebo_ros roscpp std_msgs
  #DEPENDS system_lib
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIR}
)

add_library(gazebo_model_state_timestamp_patch SHARED src/gazebo_model_state_timestamp_patch.cpp)
target_link_libraries(gazebo_model_state_timestamp_patch ${GAZEBO_LIBRARIES})
