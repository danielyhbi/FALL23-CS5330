cmake_minimum_required(VERSION 3.10)
project(project4-danielbi)

# Set c++ standard to c++11
set (CMAKE_CXX_STANDARD 11)

# output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

find_package(OpenCV REQUIRED)

include_directories(
    # Mac Intel
    # /usr/local/include/opencv4
    
    # Mac M1+
    /opt/homebrew/Cellar/opencv/4.8.1_4/include/opencv4
    ../include

    ${CMAKE_SOURCE_DIR}/src/features
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    # Mac Intel
    # /usr/local/lib/opencv4/3rdparty
    # /usr/local/lib/

    # Mac M1+
    /opt/homebrew/Cellar/opencv/4.8.1_4/lib/opencv4/3rdparty
    /opt/homebrew/lib/
)

set(FEATURES
    src/augmentedRealityDemo.cpp
)

set(FEATURES_HARRIS
    src/harrisCorner.cpp
)

add_executable(project4-danielbi-find-images ${FEATURES})
add_executable(project4-danielbi-harris-corner ${FEATURES_HARRIS})

target_link_libraries(project4-danielbi-find-images
    ${OpenCV_LIBS}
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
)

target_link_libraries(project4-danielbi-harris-corner
    ${OpenCV_LIBS}
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
