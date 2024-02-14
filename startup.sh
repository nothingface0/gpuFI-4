#!/bin/bash
# Source this script to run the necessary setup for gpuFI/GPGPU-Sim execution
env_type=${1:-release}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.0 # For supporting register usage of latest cards
export CUDA_INSTALL_PATH=/usr/local/cuda-4.2/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
if [ "$env_type" != "release" ] && [ "$env_type" != "debug" ]; then
    echo "Environment type must be \"release\" or \"debug\""
    return
fi
echo "Setting environment to $env_type"
source setup_environment $env_type
printenv >$SCRIPT_DIR/.env # To be used by VSCode debugger
echo PTX_SIM_DEBUG=3 >>$SCRIPT_DIR/.env
# If c++ std lib debug symbols are needed
#echo LD_PRELOAD="/usr/lib/x86_64-linux-gnu/debug/libstdc++.so.6" >>$SCRIPT_DIR/.env
