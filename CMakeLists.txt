cmake_minimum_required(VERSION 3.20)
project(OpenV6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(OPENV6_NATIVE "Build with -march=native (not portable)" OFF)

find_package(Threads REQUIRED)

set(OPENV6_SOURCES
    src/base64.cpp
    src/fft.cpp
    src/frame_driver.cpp
    src/frame_synth.cpp
    src/g2p.cpp
    src/json.cpp
    src/local_voice.cpp
    src/md5.cpp
    src/model_file.cpp
    src/networks.cpp
    src/parallel.cpp
    src/phoneme_conversion.cpp
    src/phonemizer.cpp
    src/pitch_model.cpp
    src/score_features.cpp
    src/speakers.cpp
    src/speex_resampler.cpp
    src/synthesizer.cpp
    src/timing.cpp
    src/voice_package.cpp
    src/vpit_features.cpp
    src/vsm_pitch.cpp
    src/vtmg_model.cpp
)

set(OPENV6_SPEAKERS_JSON ${CMAKE_CURRENT_SOURCE_DIR}/data/V6Speakers.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${OPENV6_SPEAKERS_JSON})
file(READ ${OPENV6_SPEAKERS_JSON} OPENV6_SPEAKERS_TEXT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/speakers_data.cpp
    "extern const char* const openv6_speakers_json;\nconst char* const openv6_speakers_json = R\"openv6json(${OPENV6_SPEAKERS_TEXT})openv6json\";\n")
list(APPEND OPENV6_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/speakers_data.cpp)

add_library(openv6 STATIC ${OPENV6_SOURCES})
target_include_directories(openv6 PUBLIC include)
target_link_libraries(openv6 PUBLIC Threads::Threads)
target_compile_options(openv6 PUBLIC -Wall -Wextra -ffp-contract=off)
if(OPENV6_NATIVE)
    target_compile_options(openv6 PUBLIC -march=native)
endif()

add_executable(openv6_render tools/openv6_render.cpp)
target_link_libraries(openv6_render PRIVATE openv6)
target_compile_definitions(openv6_render PRIVATE OPENV6_NATIVE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/native")

enable_testing()
add_subdirectory(tests)
