cmake_minimum_required(VERSION 3.8)
project(gicp_mapper)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=undefined -O0 -g")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

# ── Dependencias ──────────────────────────────────────────────────────────────
find_package(ament_cmake        REQUIRED)
find_package(rclcpp             REQUIRED)
find_package(sensor_msgs        REQUIRED)
find_package(geometry_msgs      REQUIRED)
find_package(nav_msgs           REQUIRED)
find_package(vision_msgs        REQUIRED)
find_package(std_srvs           REQUIRED)
find_package(tf2                REQUIRED)
find_package(tf2_ros            REQUIRED)
find_package(tf2_eigen          REQUIRED)
find_package(tf2_geometry_msgs  REQUIRED)
find_package(tf2_sensor_msgs    REQUIRED)
find_package(pcl_conversions    REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io features search kdtree)
find_package(laser_geometry     REQUIRED)
find_package(fast_gicp          REQUIRED)

find_package(std_msgs REQUIRED)

find_package(Eigen3             REQUIRED)
find_package(OpenMP             REQUIRED)
find_package(segmentation_msgs            REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(octomap QUIET)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NDTCell.msg"
  "msg/NDTCloud.msg"
  DEPENDENCIES std_msgs geometry_msgs
)


link_directories(
  ${CMAKE_PREFIX_PATH}/lib
)

# ── Ejecutable ────────────────────────────────────────────────────────────────
add_executable(gicp_mapper_node
  src/gicp_mapper_node.cpp
)

add_executable(ndt_input_scan
  src/ndt_input_scan.cpp
)


add_executable(ndt_matcher_node
  src/ndt_matcher_node.cpp
)

# add_executable(vgicp_mapper_node
#   src/vgicp_mapper_node2.cpp
# )

# add_executable(voxelized_mapper
#   src/new_voxelized_mapper_node.cpp
# )

# target_link_libraries(${target}
#   ${PCL_LIBRARIES}
# )
# Lista de ejecutables
set(MAPPER_TARGETS
  gicp_mapper_node
  ndt_input_scan
  ndt_matcher_node
  # vgicp_mapper_node
  # voxelized_mapper
)

# Includes comunes
foreach(target ${MAPPER_TARGETS})

  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
      ${PCL_INCLUDE_DIRS}
      ${EIGEN3_INCLUDE_DIRS}
  )

  target_link_libraries(${target}
    ${PCL_LIBRARIES}
    ${FAST_GICP_LIB}
    OpenMP::OpenMP_CXX
  )

  ament_target_dependencies(${target}
    rclcpp 
    sensor_msgs 
    geometry_msgs 
    nav_msgs 
    std_srvs
    tf2 
    tf2_ros 
    tf2_eigen 
    tf2_geometry_msgs 
    tf2_sensor_msgs
    pcl_conversions 
    laser_geometry 
    fast_gicp 
    segmentation_msgs 
    std_msgs
  )

  rosidl_target_interfaces(${target}
    ${PROJECT_NAME}
    "rosidl_typesupport_cpp"
  )

  target_link_libraries(${target}
    ${PCL_LIBRARIES}
    ${FAST_GICP_LIB}
    OpenMP::OpenMP_CXX
  )

  ament_target_dependencies(${target}
    rclcpp sensor_msgs geometry_msgs nav_msgs std_srvs
    tf2 tf2_ros tf2_eigen tf2_geometry_msgs tf2_sensor_msgs
    pcl_conversions laser_geometry fast_gicp segmentation_msgs std_msgs
  )
endforeach()

# ── Instalar ──────────────────────────────────────────────────────────────────
install(TARGETS gicp_mapper_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS ndt_input_scan DESTINATION lib/${PROJECT_NAME})
install(TARGETS ndt_matcher_node DESTINATION lib/${PROJECT_NAME})

# install(TARGETS vgicp_mapper_node DESTINATION lib/${PROJECT_NAME})
# install(TARGETS voxelized_mapper DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/       DESTINATION include)
install(DIRECTORY launch config  DESTINATION share/${PROJECT_NAME} OPTIONAL)

ament_package()
