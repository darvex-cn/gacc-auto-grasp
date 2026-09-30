cmake_minimum_required(VERSION 3.8)
project(grasp_coordinator)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
