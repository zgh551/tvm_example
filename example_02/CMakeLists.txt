cmake_minimum_required(VERSION 3.16)

project(mnist_module)
## Set current version and minimal executable version.
set(MAJOR_VERSION 1)
set(MISOR_VERSION 1)
set(REVISION_VERSION 2)

set(MNIST_VERSION ${MAJOR_VERSION}.${MISOR_VERSION}.${REVISION_VERSION})

add_definitions(-DMNIST_VERSION=\"${MNIST_VERSION}\")
message(STATUS "Software Version: ${MNIST_VERSION}")

set(TARGET_DEVICE_TYPE CPU CACHE STRING "the target device type")
set_property(CACHE TARGET_DEVICE_TYPE PROPERTY STRINGS CPU OpenCL)

if(TARGET_DEVICE_TYPE MATCHES CPU)
    add_definitions(-DCPU)
    message(STATUS "Target Device Type: CPU")
elseif(TARGET_DEVICE_TYPE MATCHES OpenCL)
    add_definitions(-DOpenCL)
    message(STATUS "Target Device Type: OpenCL")
else()
    message(FATAL_ERROR "Invalid Target Device Type: " ${TARGET_DEVICE_TYPE})
endif()

# find opencv lib
find_package(OpenCV REQUIRED)

# set(TVM_HOME "pathto/tvm")
include_directories(
    $ENV{TVM_HOME}/3rdparty/dlpack/include
    $ENV{TVM_HOME}/3rdparty/dmlc-core/include
    $ENV{TVM_HOME}/include
    ${OpenCV_INCLUDE_DIRS}
)
message(STATUS ${OpenCV_INCLUDE_DIRS})
#-Wall -Werror
add_compile_options(-std=c++14 -O2)

## set the link lib location
## link the dynamic lib
set(MACHINE_NAME x86_64 CACHE STRING "the machine name")
set_property(CACHE MACHINE_NAME PROPERTY STRINGS x86_64 aarch64)

if (${MACHINE_NAME} MATCHES x86_64) 
    message(STATUS "Link Dynamic Lib Type: " ${MACHINE_NAME})
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64)
elseif (${MACHINE_NAME} MATCHES aarch64)
    message(STATUS "Link Dynamic Lib Type: " ${MACHINE_NAME})
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/aarch64)
else ()
    message(FATAL_ERROR "Invalid Machine name: " ${MACHINE_NAME})
endif()

add_executable(mobilenet_run mobilenet.cc)
#add_executable(${MACHINE_NAME}_run main.cc)

target_link_libraries(mobilenet_run tvm_runtime ${OpenCV_LIBS} pthread dl)
