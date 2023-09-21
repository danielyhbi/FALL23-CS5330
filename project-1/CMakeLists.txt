cmake_minimum_required(VERSION 3.10)
project(project1-danielbi)

# Set c++ standard to c++11
set (CMAKE_CXX_STANDARD 11)

# output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/compiled)

include_directories(
    /usr/local/include/opencv4
    ../include
)

link_directories(
    /usr/local/lib/opencv4/3rdparty
    /usr/local/lib/
)

set(SOURCES_vid
    vidDisplay.cpp
    filters.cpp
)

add_executable(project1-danielbi-imgDisplay imgDisplay.cpp filters.cpp filterMatrices.h)
add_executable(project1-danielbi-vidDisplay vidDisplay.cpp filters.cpp filterMatrices.h)

target_link_libraries(project1-danielbi-imgDisplay
    -ltiff
    -lpng
    -ljpeg
    -llapack
    -lblas
    -lz
    -ljasper
    -lwebp
    -IlmImf
    -lopencv_core
    -lopencv_highgui
    # -lopencv_video
    # -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_imgproc
    # -lopencv_objdetect
)

target_link_libraries(project1-danielbi-vidDisplay
    -ltiff
    -lpng
    -ljpeg
    -llapack
    -lblas
    -lz
    -ljasper
    -lwebp
    -IlmImf
    -lopencv_core
    -lopencv_highgui
    -lopencv_video
    -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_imgproc
    # lopencv_objdetect
)

link_libraries(
    -framework AVFoundation
    -framework CoreMedia
    -framework CoreVideo
    -framework CoreServices
    -framework CoreGraphics
    -framework AppKit
    -framework OpenCL
)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
