
cmake_minimum_required(VERSION 3.16)
# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Avoid BUILD_SHARED_LIBS getting overridden by an option() in ccd.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(backprop_tools)

set(CMAKE_CXX_STANDARD 17)

message("Build type: ${CMAKE_BUILD_TYPE}")