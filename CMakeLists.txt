cmake_minimum_required(VERSION 3.18)

project(neuralnetwork LANGUAGES CXX)

option(USE_CUDA "Enable CUDA support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(neural_net ${SOURCES})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(neural_net PRIVATE -Wall -Wextra -O2)
elseif (MSVC)
    target_compile_options(neural_net PRIVATE /W4)
endif()

if(USE_CUDA)
    enable_language(CUDA)

    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/cuda/*.cu
    )

    target_sources(neural_net PRIVATE ${CUDA_SOURCES})

    set_target_properties(neural_net PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    target_compile_definitions(neural_net PRIVATE USE_CUDA)
endif()