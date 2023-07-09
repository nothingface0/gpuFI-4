#!/bin/bash
# Source this script in order to setup for gpuFI/GPGPU-Sim execution
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.0 # For supporting register usage of latest cards
export CUDA_INSTALL_PATH=/usr/local/cuda-4.2/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
source setup_environment debug	# Keep it debug for now
printenv > $SCRIPT_DIR/.env		# To be used by VSCode debugger
echo PTX_SIM_DEBUG=3 >> $SCRIPT_DIR/.env
echo LD_PRELOAD="/usr/lib/x86_64-linux-gnu/debug/libstdc++.so.6" >> $SCRIPT_DIR/.env