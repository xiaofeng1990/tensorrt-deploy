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


set(MAJOR_VERSION     1)
set(MINOR_VERSION     0)
set(REVISION_VERSION  4)

string(TIMESTAMP XF_BUILD_DATE "%Y%m%d%H")
configure_file("${CMAKE_SOURCE_DIR}/master/template/versin.h.in"
    "${CMAKE_BINARY_DIR}/include/version.h")
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# aux_source_directory(${PROJECT_SOURCE_DIR}/api             API_SRC_FILES)
set(TARGET_NAME xf_deploy)

add_executable(${TARGET_NAME}
    main.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE ${LINKS_LIBS})

# install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})