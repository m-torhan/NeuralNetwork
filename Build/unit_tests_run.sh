#!/bin/bash
if [[ -z "$1" ]]; then
    FILTER="*"
else
    FILTER="$1"
fi
cd ../tests/unit_tests/Build/Linux/Release/
./NeuralNetwork_unit_tests --gtest_filter="$FILTER"