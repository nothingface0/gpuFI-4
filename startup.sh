#!/bin/bash
env_type=${1:-release}
# Source this script in order to setup for gpuFI/GPGPU-Sim execution
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.0 # For supporting register usage of latest cards
export CUDA_INSTALL_PATH=/usr/local/cuda-4.2/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
echo "Setting environment to $env_type"
source setup_environment $env_type
printenv >$SCRIPT_DIR/.env # To be used by VSCode debugger
# Does not work, didn't have time to look into it.
# echo WYS_EXEC_PATH=$HOME/Documents/workspace/cuda-tests/kernels/test_if/bin/test_if >> $SCRIPT_DIR/.env
# echo WYS_EXEC_NAME=test_if >> $SCRIPT_DIR/.env
echo PTX_SIM_DEBUG=3 >>$SCRIPT_DIR/.env
#echo LD_PRELOAD="/usr/lib/x86_64-linux-gnu/debug/libstdc++.so.6" >>$SCRIPT_DIR/.env
