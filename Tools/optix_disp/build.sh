#!/bin/bash
# Builds the march port: build/optix_disp_cpu (CPU check) and, with CUDA and OptiX, build/optix_disp and its OptiX IR (sm_86).
set -e; cd "$(dirname "$0")"; B=../../build; OPTIX_HOME=${OPTIX_HOME:-/opt/optix}; CUDA_HOME=${CUDA_HOME:-/opt/cuda}
g++ -O2 -std=c++17 -Wall -o $B/optix_disp_cpu cpu_render.cpp -lpthread
[ -f $OPTIX_HOME/include/optix.h ] || { echo "no OptiX at $OPTIX_HOME: CPU check only"; exit 0; }
$CUDA_HOME/bin/nvcc -optix-ir -std=c++17 -arch=${ARCH:-sm_86} -I$OPTIX_HOME/include -o $B/optix_disp.optixir optix_disp.cu
g++ -O2 -std=c++17 -Wall -I$OPTIX_HOME/include -I$CUDA_HOME/include -o $B/optix_disp optix_disp.cpp -L$CUDA_HOME/lib64 -lcuda -ldl
