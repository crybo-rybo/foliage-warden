cmake_minimum_required(VERSION 3.22)

project(foliage_warden_cpp_conformance LANGUAGES CXX)

set(FOLIAGE_WARDEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../cpp" foliage_warden_cpp)

add_executable(foliage_warden_cpp_conformance probe.cpp)
target_link_libraries(foliage_warden_cpp_conformance PRIVATE foliage::policy)
target_compile_features(foliage_warden_cpp_conformance PRIVATE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(foliage_warden_cpp_conformance PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
endif()
