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

# Get a list of valid GPU_IDs to use
_get_valid_gpu_ids() {
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    ls "$SCRIPT_DIR/configs/tested-cfgs"
}
