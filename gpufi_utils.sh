#!/bin/bash

# gpuFI Utils: helper functions for other scripts.
# This script is meant to be sourced by other scripts, not run directly.

# The messages expected to be found inside each simulator's execution log
# and signify whether the final data are the same with the golden execution of
# the benchmark.
_SUCCESS_MSG='Test PASSED'
_FAILED_MSG='Test FAILED'

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
    path_to_gpgpu_sim_config=${1?no gpgpusim config file supplied}
    executable_path=${2?no executable path supplied}
    executable_args=${3:- }
    if [ ! -f "$path_to_gpgpu_sim_config" ]; then
        return
    fi
    if [ ! -f "$executable_path" ]; then
        return
    fi
    # Don't take gpufi_run_id into account.
    echo -n "$(grep -Ev '\-gpufi_run_id.+' $path_to_gpgpu_sim_config)$(cat $executable_path)${executable_args}" | md5sum | awk '{print $1}'
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

# Examine a given execution log file and assess the execution results.
# Exports the resulting variables.
# Also requires the _TOTAL_CYCLES that the executable is expected to run for and the
# _L1I_CACHE_TOTAL_MISSES (calculated during the executable_analysis script)
_examine_log_file() {
    log_file=$1
    total_cycles=${2?Expected total cycles not specified}
    l1i_cache_total_misses=${3?L1I Cache expected misses not specified}

    grep -iq "${_SUCCESS_MSG}" "$log_file" && success_msg_grep=0 || success_msg_grep=1
    grep -i "${_CYCLES_MSG}" "$log_file" | tail -1 | grep -q "${total_cycles}" && cycles_grep=0 || cycles_grep=1
    grep -iq "${_FAILED_MSG}" "$log_file" && failed_msg_grep=0 || failed_msg_grep=1
    grep -iqE "(syntax error)|(parse error)" "$log_file" && syntax_error_msg_grep=0 || syntax_error_msg_grep=1
    grep -iq "gpuFI: Tag before" "$log_file" && tag_bitflip_grep=0 || tag_bitflip_grep=1
    grep -iq "gpuFI: Resulting injected instruction" "$log_file" && data_bitflip_grep=0 || data_bitflip_grep=1
    grep -iq "gpuFI: False L1I cache hit due to tag" "$log_file" && false_l1i_hit_grep=0 || false_l1i_hit_grep=1
    grep -i "L1I_total_cache_misses" "$log_file" | tail -1 | grep -q "${l1i_cache_total_misses}" && different_l1i_misses=0 || different_l1i_misses=1

    export success_msg_grep
    export cycles_grep
    export failed_msg_grep
    export syntax_error_msg_grep
    export tag_bitflip_grep
    export data_bitflip_grep
    export false_l1i_hit_grep
    export different_l1i_misses
}

# Updates the results.csv file, given the variables exported from the _examine_log_file function.
_update_csv_file() {
    csv_results_path="$(_get_gpufi_analysis_path)/results"
    mkdir -p "$csv_results_path"

    csv_file_path="$csv_results_path/results.csv"
    run_id=$1
    # Turn 0 to 1 and the opposite, it's clearer if each flag is "1" if the event it's
    # describing happened.
    success_msg_grep=$2
    success_msg_grep=$((success_msg_grep ^ 1))
    cycles_grep=$3
    cycles_grep=$((cycles_grep ^ 1))
    failed_msg_grep=$4
    failed_msg_grep=$((failed_msg_grep ^ 1))
    syntax_error_msg_grep=$5
    syntax_error_msg_grep=$((syntax_error_msg_grep ^ 1))
    tag_bitflip_grep=$6
    tag_bitflip_grep=$((tag_bitflip_grep ^ 1))
    l1i_data_bitflip_grep=$7
    l1i_data_bitflip_grep=$((l1i_data_bitflip_grep ^ 1))
    false_l1i_hit_grep=$8
    false_l1i_hit_grep=$((false_l1i_hit_grep ^ 1))
    different_l1i_misses=$8
    different_l1i_misses=$((different_l1i_misses ^ 1))

    # Flag to control whether we should check if the run_id exists in the csv file, in order
    # to replace it or not.
    replace_exising_run=${9:-1}

    if [ ! -f "$csv_file_path" ]; then
        echo "run_id,success,same_cycles,failed,syntax_error,tag_bitflip,l1i_data_bitflip,false_l1i_hit,different_l1i_misses" >"$csv_file_path"
    fi
    echo "Updating results in $csv_file_path"
    # gpuFI TODO: Check whether run_id already exists, compare results, should be the same!
    if [ $replace_exising_run -ne 0 ]; then
        if grep "${run_id}" "$(_get_gpufi_analysis_path)/results/results.csv"; then
            echo "$run_id already exists in results.csv, updating existing entry"
            sed -Ei "s/^${run_id}(,[01]){8}/${run_id},${success_msg_grep},${cycles_grep},${failed_msg_grep},${syntax_error_msg_grep},${tag_bitflip_grep},${l1i_data_bitflip_grep},${false_l1i_hit_grep},${different_l1i_misses}/" "$(_get_gpufi_analysis_path)/results/results.csv"
        fi
    else
        echo "${run_id},${success_msg_grep},${cycles_grep},${failed_msg_grep},${syntax_error_msg_grep},${tag_bitflip_grep},${l1i_data_bitflip_grep},${false_l1i_hit_grep},${different_l1i_misses}" >>"$csv_file_path"
    fi

}
