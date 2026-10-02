cmake_minimum_required(VERSION 3.16)
project(NeuraCPP LANGUAGES CXX)

# Require C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for speed if no build type is given
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler warnings + optimization
if(MSVC)
  add_compile_options(/W4)
  # /O2 is already implied by Release; keep explicit for clarity
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()

# Header-only engine library (all layers live in include/)
add_library(neuracpp_headers INTERFACE)
target_include_directories(neuracpp_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# OpenMP for CPU parallelism (20-core machines sit idle otherwise).
# Falls back to single-threaded if not found.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found, enabling multi-threaded training")
else()
  message(STATUS "OpenMP NOT found, single-threaded fallback")
endif()

# Main training executable
add_executable(neuracpp src/main.cpp)
target_link_libraries(neuracpp PRIVATE neuracpp_headers)
if(OpenMP_CXX_FOUND)
  target_link_libraries(neuracpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Copy helper scripts / docs expectations: warn if data/ is missing at runtime
# (dataset itself is downloaded on demand, never committed)
message(STATUS "NeuraCPP | C++${CMAKE_CXX_STANDARD} | ${CMAKE_BUILD_TYPE}")

# Install rule: `cmake --install build --prefix <dir>`
install(TARGETS neuracpp RUNTIME DESTINATION bin)

# Optional: CTest smoke test (compiles main only, dataset-dependent run is manual)
include(CTest)
