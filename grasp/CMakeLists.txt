cmake_minimum_required(VERSION 3.0.2)
project(grasp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

add_service_files(
  FILES
  GetPosition.srv
  GetVelocity.srv
  SetPosition.srv
  ResetMotor.srv
  SetCurrent.srv 
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/aruco_pub.py
  scripts/main_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)