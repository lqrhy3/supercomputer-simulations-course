#!/bin/bash
g++ -std=c++17 main.cpp -o main -fopenmp
OMP_NUM_THREAD=$1 ./main
