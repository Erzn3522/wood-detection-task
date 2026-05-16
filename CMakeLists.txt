cmake_minimum_required(VERSION 3.20)
project(wood_knot_detector VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)

# ---------------------------------------------------------------------------
# Header-only deps via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.0
)
FetchContent_MakeAvailable(argparse)

# ---------------------------------------------------------------------------
# ONNX Runtime  (set -DONNXRUNTIME_ROOT=<path/to/onnxruntime>)
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of extracted ONNX Runtime prebuilt tarball")

if(ONNXRUNTIME_ROOT STREQUAL "")
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set.\n"
        "Download the prebuilt tarball from https://github.com/microsoft/onnxruntime/releases\n"
        "and pass -DONNXRUNTIME_ROOT=<extracted-dir> to cmake.")
endif()

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED
)
find_library(ORT_LIB onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED
)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(wood_knot_detector
    src/main.cpp
    src/frame_loader.cpp
    src/io.cpp
    src/yolo_session.cpp
    src/board_inference.cpp
    src/evaluator.cpp
    src/visualizer.cpp
)

target_include_directories(wood_knot_detector PRIVATE
    include
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(wood_knot_detector PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIB}
    nlohmann_json::nlohmann_json
    argparse::argparse
)

target_compile_options(wood_knot_detector PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS wood_knot_detector DESTINATION bin)
