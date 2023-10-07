cmake_minimum_required(VERSION 3.10)
project(project2-danielbi)

# Set c++ standard to c++11
set (CMAKE_CXX_STANDARD 11)

# output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

include_directories(
    # Mac Intel
    # /usr/local/include/opencv4
    
    # Mac M1+
    /opt/homebrew/Cellar/opencv/4.8.0_7/include/opencv4
    ../include

    ${CMAKE_SOURCE_DIR}/src/features
    ${CMAKE_SOURCE_DIR}/include
)

link_directories(
    # Mac Intel
    # /usr/local/lib/opencv4/3rdparty
    # /usr/local/lib/

    # Mac M1+
    /opt/homebrew/Cellar/opencv/4.8.0_7/lib/opencv4/3rdparty
    /opt/homebrew/lib/
)

set(FEATURES
    src/findImages.cpp
    src/features/feature.h
    src/features/baseline_matching.cpp
    src/features/histogram_matching.cpp
    src/features/multiHistMatching.cpp
    src/features/feature.cpp
    src/features/multiHistTextureMatch.cpp
    src/features/custom1.cpp
    src/features/custom2.cpp
    src/features/custom3_roads.cpp
    include/csv_util.h
)

add_executable(project1-danielbi-find-images ${FEATURES})

target_link_libraries(project1-danielbi-find-images
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
