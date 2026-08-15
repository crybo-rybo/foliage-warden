cmake_minimum_required(VERSION 3.22)

project(foliage_warden_policy VERSION 0.1.0 LANGUAGES CXX)

option(FOLIAGE_WARDEN_BUILD_TESTS "Build the dependency-free policy tests" ON)

add_library(foliage_warden_policy
    src/mock_actuator.cpp
    src/policy_engine.cpp
    src/policy_types.cpp
)
add_library(foliage::policy ALIAS foliage_warden_policy)

target_compile_features(foliage_warden_policy PUBLIC cxx_std_20)
target_include_directories(foliage_warden_policy
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(foliage_warden_policy PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()

if(FOLIAGE_WARDEN_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(foliage_warden_policy_tests tests/policy_tests.cpp)
    target_link_libraries(foliage_warden_policy_tests PRIVATE foliage::policy)
    target_compile_features(foliage_warden_policy_tests PRIVATE cxx_std_20)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(foliage_warden_policy_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()

    add_test(NAME foliage_warden_policy_tests COMMAND foliage_warden_policy_tests)
endif()
