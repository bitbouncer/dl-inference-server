cmake_minimum_required(VERSION 2.8.11)

SET(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(BUILD_STATIC_LIBS "enable static linking support" ON)
option(BUILD_SHARED_LIBS "enable shared linking support" OFF)

#GLOBAL DEFINITIONS
add_definitions(-DRAPIDJSON_HAS_STDSTRING)
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wpedantic -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_REENTRANT -DEXTERNAL_LOCKS -DMULTITHREAD)
add_definitions(-fPIC)
add_definitions(-Wno-deprecated)
add_definitions(-g)

include_directories(${CMAKE_SOURCE_DIR})
link_directories(${DEV_LIBRARY_PATH})

#INCLUDE(FindProtobuf)
#FIND_PACKAGE(Protobuf REQUIRED)
#INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
#PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER src/core/grpc_service.proto)
#ADD_LIBRARY(proto ${PROTO_HEADER} ${PROTO_SRC})

find_package(OpenCV REQUIRED)

#highgui core imgproc

FILE(GLOB LIB_SRCS
        src/core/*.cc
        src/core/*.h
        src/clients/c++/request.cc
        src/clients/c++/request.h
        )

if (BUILD_SHARED_LIBS)
    ADD_LIBRARY(dl-inference SHARED ${LIB_SRCS} ${PROTO_HEADER} ${PROTO_SRC})
    INSTALL(TARGETS dl-inference
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            )
endif ()

if (BUILD_STATIC_LIBS)
    ADD_LIBRARY(dl-inference_s STATIC ${LIB_SRCS} ${PROTO_HEADER} ${PROTO_SRC})
    INSTALL(TARGETS dl-inference_s
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            )
endif ()

SET(EXT_LIBS_SHARED dl-inference  grpc++ protobuf curl glog crypto ssl pthread)
SET(EXT_LIBS_STATIC dl-inference_s grpc++ protobuf curl glog crypto ssl pthread)

include_directories(${CMAKE_SOURCE_DIR})
link_directories(${DEV_LIBRARY_PATH})


add_executable(perf_client src/clients/c++/perf_client.cc)

if (LINK_SHARED)
    target_link_libraries(perf_client ${EXT_LIBS_SHARED})
else ()
    target_link_libraries(perf_client ${EXT_LIBS_STATIC})
endif ()

INSTALL(TARGETS perf_client RUNTIME DESTINATION bin)



add_executable(image_client src/clients/c++/image_client.cc)

if (LINK_SHARED)
    target_link_libraries(image_client ${OpenCV_LIBRARIES} ${EXT_LIBS_SHARED})
else ()
    target_link_libraries(image_client ${OpenCV_LIBRARIES} ${EXT_LIBS_STATIC})
endif ()

INSTALL(TARGETS image_client RUNTIME DESTINATION bin)








