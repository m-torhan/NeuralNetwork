#!/bin/bash
cmake -G"Ninja" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=$1 -DCMAKE_BUILD_MODE=$2 .
cmake --build .