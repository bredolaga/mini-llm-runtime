cmake_minimum_required(VERSION 3.20)
project(mini_llm_runtime)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    safetensors_cpp
    GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp.git
    GIT_TAG main
)

FetchContent_GetProperties(safetensors_cpp)

if(NOT safetensors_cpp_POPULATED)
    FetchContent_Populate(safetensors_cpp)
endif()

add_executable(mini_llm
    src/main.cpp
    src/ops.cpp
    src/attention.cpp
    src/mlp.cpp
    src/model.cpp
    src/loader.cpp
    src/tokenizer.cpp
)

target_include_directories(mini_llm PRIVATE
    include
    ${safetensors_cpp_SOURCE_DIR}
)
