#!/bin/bash

# gpuFI Utils: helper functions for other scripts.
# This script is meant to be sourced by other scripts, not run directly.

# Function to sanitize args to a folder name
# From here: https://stackoverflow.com/a/44811468/6562491
# echoes "null" if no input given.
_sanitize_string() {
    local s="${*:-null}"     # receive input in first argument
    s="${s//[^[:alnum:]]/-}" # replace all non-alnum characters to -
    s="${s//+(-)/-}"         # convert multiple - to single -
    s="${s/#-/}"             # remove - from start
    s="${s/%-/}"             # remove - from end
    echo "${s,,}"            # convert to lowercase
}

# Cehck if GPU_ID is valid
_is_gpu_id_valid() {
    gpu_id=$1
    if [ -z "$gpu_id" ]; then
        return 1
    fi

    if [ -f "$(_get_gpgpusim_config_path_from_gpu_id $gpu_id)" ]; then
        return 0
    fi
    return 1
}

_get_timestamp() {
    echo "[$(printf '%(%Y-%m-%d %H:%M:%S)T' -1)]"
}

# Create a unique id for a specific run after it's complete, based on
# the contents of the gpgpusim, the contents of
# the executable and the args it run with.
_calculate_md5_hash() {
    path_to_gpgpu_sim_config=${1-:./gpgpusim.config}
    executable_path=${3?no executable path supplied}
    executable_args=${4:- }
    if [ ! -f "$path_to_gpgpu_sim_config" ]; then
        return
    fi
    if [ ! -f "$executable_path" ]; then
        return
    fi
    echo -n "$(cat $path_to_gpgpu_sim_config)$(cat $executable_path)${executable_args}" | md5sum | awk '{print $1}'
}

# Given a GPU_ID, constructs the path to its gpgpusim.config file.
# The path is not guaranteed to exist.
_get_gpgpusim_config_path_from_gpu_id() {
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    echo "$SCRIPT_DIR/configs/tested-cfgs/$gpu_id/gpgpusim.config"
}

# Get the path to a unique directory in the same directory the executable is in
# which is identified by the GPU_ID and the arguments the executable is run with,
# after sanitization.
_get_gpufi_analysis_path() {
    echo "$(dirname $CUDA_EXECUTABLE_PATH)/.gpufi/$GPU_ID/$(_sanitize_string $CUDA_EXECUTABLE_ARGS)"
}

# Copies the selected GPU_ID's gpgpusim.config file in the current directory
# Assumes a _GPGPU_SIM_CONFIG_PATH variable in your script which holds
# a path to the gpgpusim.config.
_copy_gpgpusim_config() {
    gpu_id=$1
    # Current script's absolute dir. All other paths are calculated relative to it.
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

    if [ -z "$_GPGPU_SIM_CONFIG_PATH" ]; then
        # A path to a gpgpusim.config has not been provided.
        # Get it from the tested configs.
        original_config=$(_get_gpgpusim_config_path_from_gpu_id "$gpu_id")
        cp "$original_config" "$SCRIPT_DIR/gpgpusim.config"
        _GPGPU_SIM_CONFIG_PATH="$SCRIPT_DIR/gpgpusim.config"
    else
        cp "$_GPGPU_SIM_CONFIG_PATH" "$SCRIPT_DIR/gpgpusim.config"
    fi
}
