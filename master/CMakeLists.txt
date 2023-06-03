CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

enable_testing()

project(xf_deploy CXX C)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "output directory")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "[0-9].[0-9].[0-9]*"
        WORKING_DIRECTORY "${${PROJECT_NAME}_SOURCE_DIR}"
        OUTPUT_VARIABLE XT_GIT_ONELINE
        RESULT_VARIABLE GET_RESULT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GET_RESULT EQUAL 0)
        set(XT_GIT_ONELINE "unknow")
    endif()
else()
    set(XT_GIT_ONELINE "unknow")
endif()

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)
 
message(STATUS "glib-2.0 library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${GLIB_INCLUDE_DIRS}")
message(STATUS "    lib path: ${GLIB_LIBRARY_DIRS}")
message(STATUS "    lib: ${GLIB_LIBRARIES}")
include_directories(${GLIB_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS})
set(LINKS_LIBS ${LINKS_LIBS} ${GLIB_LIBRARIES})

set(MAJOR_VERSION     1)
set(MINOR_VERSION     0)
set(REVISION_VERSION  4)

string(TIMESTAMP XF_BUILD_DATE "%Y%m%d%H")
configure_file("${CMAKE_SOURCE_DIR}/master/template/versin.h.in"
    "${CMAKE_BINARY_DIR}/include/version.h")
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(.)
include_directories(..)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")



find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "    lib path: ${CUDA_LIBRARIES}")
    message(STATUS "    install path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "    CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc) 
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(LINKS_LIBS ${LINKS_LIBS} ${CUDA_LIBRARIES} -lcuda)
else(CUDA_FOUND)
    message(FATA_ERROR "Do not find CUDA")
endif(CUDA_FOUND)


find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer)
message(STATUS "Found TensorRT lib at ${TENSORRT_LIBRARY_INFER}")
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser)
message(STATUS "Found TensorRT lib at ${TENSORRT_LIBRARY_ONNXPARSER}")
include_directories(${TENSORRT_INCLUDE_DIR})
set(LINKS_LIBS ${LINKS_LIBS} ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})


aux_source_directory(${PROJECT_SOURCE_DIR}/common       COMMON_SRC_FILES)
aux_source_directory(${PROJECT_SOURCE_DIR}/inference    INFER_SRC_FILES)
aux_source_directory(${PROJECT_SOURCE_DIR}/tensorrt     TRT_SRC_FILES)

set(TARGET_NAME xf_deploy)

cuda_add_executable(${TARGET_NAME}
    main.cpp config_env.cpp
    ${COMMON_SRC_FILES}
    ${INFER_SRC_FILES}
    ${TRT_SRC_FILES}
)

target_link_libraries(${TARGET_NAME} ${LINKS_LIBS})

# install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})