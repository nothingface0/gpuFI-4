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
