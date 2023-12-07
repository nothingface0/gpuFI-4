#!/bin/bash

# Function to sanitize args to a folder name
# From here: https://stackoverflow.com/a/44811468/6562491
# echoes "null" if no input given.
_sanitize() {
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
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    for valid_gpu_id_directory in $SCRIPT_DIR/configs/tested-cfgs/*; do
        if [ "$gpu_id" = "$(basename $valid_gpu_id_directory)" ]; then
            return 0
        fi
    done
    return 1
}

_get_timestamp() {
    echo "[$(printf '%(%Y-%m-%d %H:%M:%S)T' -1)]"
}

# Create a unique id for a specific run after it's complete, based on
# the contents of the gpgpusim, the contents of the log file, the path to the executable and the args
# it run with.
_calculate_md5_hash() {
    path_to_gpgpu_sim_config=${1-:./gpgpusim.config}
    path_to_output_log=${2?no path to output log given}
    executable_path=${3?no executable path supplied}
    executable_args=${4:- }
    if [ ! -f "$path_to_gpgpu_sim_config" ]; then
        echo "$path_to_gpgpu_sim_config is not a valid file"
        return
    fi
    if [ ! -f "$path_to_output_log" ]; then
        echo "$path_to_output_log is not a valid file"
        return
    fi
    if [ ! -f "$executable_path" ]; then
        echo "$executable_path is not a valid file"
        return
    fi
    echo -n "$(cat $path_to_gpgpu_sim_config)$(cat $path_to_output_log)${executable_path}${executable_args}" | md5sum | awk '{print $1}'
}
