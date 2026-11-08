cmake_minimum_required(VERSION 3.18)
project(weightedcluster_core LANGUAGES CXX)

# The published package declares pybind11 as its compiled-core dependency.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_wc_core MODULE cpp/wc_core.cpp)

# Require C++17
if (TARGET _wc_core)
  target_compile_features(_wc_core PRIVATE cxx_std_17)
endif()

# On MSVC, enable parallel build and reasonable warnings
if (MSVC AND TARGET _wc_core)
  target_compile_options(_wc_core PRIVATE /W3 /EHsc)
endif()

# Ensure the built binary is named _wc_core (no lib prefix) and install only if target exists
if (TARGET _wc_core)
  set_target_properties(_wc_core PROPERTIES OUTPUT_NAME "_wc_core")
  # Install into the weightedcluster package at install time
  install(TARGETS _wc_core
          LIBRARY DESTINATION weightedcluster
          RUNTIME DESTINATION weightedcluster
          ARCHIVE DESTINATION weightedcluster)
endif()
