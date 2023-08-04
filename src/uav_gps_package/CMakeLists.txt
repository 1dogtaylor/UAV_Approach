cmake_minimum_required(VERSION 3.0.2)
project(uav_gps_package)


find_package(catkin REQUIRED COMPONENTS
  mavros_msgs
  rospy
)



catkin_package(
  CATKIN_DEPENDS rospy mavros_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
  scripts/my_python_script
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

