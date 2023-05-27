CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

project(XfTensorRTDeploy CXX C)

set(EXECUTABLE_OUTPUT_PATH   "${CMAKE_BINARY_DIR}/bin" CACHE PATH "bin output directory")
set(LIBRARY_OUTPUT_PATH      "${CMAKE_BINARY_DIR}/lib" CACHE PATH "library output directory")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# 指定lib搜索路径
set(CMAKE_INSTALL_RPATH ".:lib:output/install/lib")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_RELEASE "If ON, build release." OFF)

# set(CMAKE_BUILD_TYPE Release)

if(BUILD_RELEASE)
    set(CMAKE_BUILD_TYPE Release)
else(BUILD_RELEASE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if (UNIX)
  include(CheckFunctionExists)
  include(CheckIncludeFile)
  include(CheckSymbolExists)
  CHECK_INCLUDE_FILE(pthread.h HAVE_PTHREAD)
  CHECK_INCLUDE_FILE(malloc.h HAVE_MALLOC)
  set(LINKS_LIBS ${LINKS_LIBS} m pthread stdc++fs)
endif()

if(BUILD_RELEASE)
    set(XF_COMPILE_FLAGS "-g -O2")
else(BUILD_RELEASE)
    set(XF_COMPILE_FLAGS "-g -O0")
endif()

set(XF_COMPILE_FLAGS "${XF_COMPILE_FLAGS} -ffunction-sections -fdata-sections -funwind-tables")
set(XF_COMPILE_FLAGS "${XF_COMPILE_FLAGS} -fstack-protector-strong")
set(CMAKE_C_FLAGS    "${XF_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS  "${XF_COMPILE_FLAGS}")

add_subdirectory(master)