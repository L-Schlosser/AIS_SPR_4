cmake_minimum_required(VERSION 3.22.1)

project(bmd_go_document_processor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

# Import native libraries from llama_cpp_dart
# The actual native binaries should be in the Flutter plugin's android/src/main/jniLibs

