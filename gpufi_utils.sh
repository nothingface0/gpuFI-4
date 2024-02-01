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
    gpu_id=$1
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    echo "$SCRIPT_DIR/configs/tested-cfgs/$gpu_id/gpgpusim.config"
}

# Get the path to a unique directory in the same directory the executable is in
# which is identified by the GPU_ID and the arguments the executable is run with,
# after sanitization.
# Requires the CUDA_EXECUTABLE_PATH, GPU_ID and CUDA_EXECUTABLE_ARGS env vars to be set.
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

    grep -iq "${_SUCCESS_MSG}" "$log_file" && success_msg_grep=1 || success_msg_grep=0
    grep -i "gpu_tot_sim_cycle =" "$log_file" | tail -1 | grep -q "${total_cycles}" && cycles_grep=1 || cycles_grep=0
    grep -iq "${_FAILED_MSG}" "$log_file" && failed_msg_grep=1 || failed_msg_grep=0
    grep -iqE "(syntax error)|(parse error)" "$log_file" && syntax_error_msg_grep=1 || syntax_error_msg_grep=0
    grep -iq "gpuFI: Tag before" "$log_file" && tag_bitflip_grep=1 || tag_bitflip_grep=0
    grep -iq "gpuFI: Resulting injected instruction" "$log_file" && data_bitflip_grep=1 || data_bitflip_grep=0
    grep -iq "gpuFI: False L1I cache hit due to tag" "$log_file" && false_l1i_hit_grep=1 || false_l1i_hit_grep=0
    grep -i "L1I_total_cache_misses" "$log_file" | tail -1 | grep -q "${l1i_cache_total_misses}" && l1i_misses_grep=1 || l1i_misses_grep=0

    export success_msg_grep
    export cycles_grep
    export failed_msg_grep
    export syntax_error_msg_grep
    export tag_bitflip_grep
    export data_bitflip_grep
    export false_l1i_hit_grep
    export l1i_misses_grep
}

# Updates the results.csv file, given the variables exported from the _examine_log_file function.
_update_csv_file() {
    csv_results_path="$(_get_gpufi_analysis_path)/results"
    mkdir -p "$csv_results_path"

    csv_file_path="$csv_results_path/results.csv"
    run_id=$1

    success_msg_grep=$2
    cycles_grep=$3
    failed_msg_grep=$4
    syntax_error_msg_grep=$5
    tag_bitflip_grep=$6
    l1i_data_bitflip_grep=$7
    false_l1i_hit_grep=$8
    l1i_misses_grep=$9
    # Inverse logic for this flag, it's 1 when the misses were different
    different_l1i_misses=$((l1i_misses_grep ^ 1))

    if [ ! -f "$csv_file_path" ]; then
        echo "run_id,success,same_cycles,failed,syntax_error,tag_bitflip,l1i_data_bitflip,false_l1i_hit,different_l1i_misses" >"$csv_file_path"
    fi
    echo "Updating results in $csv_file_path"
    if grep -q "${run_id}" "$(_get_gpufi_analysis_path)/results/results.csv"; then
        echo "$run_id already exists in results.csv, updating existing entry"
        sed -Ei "s/^${run_id}(,[01]){8}/${run_id},${success_msg_grep},${cycles_grep},${failed_msg_grep},${syntax_error_msg_grep},${tag_bitflip_grep},${l1i_data_bitflip_grep},${false_l1i_hit_grep},${different_l1i_misses}/" "$(_get_gpufi_analysis_path)/results/results.csv"
    else
        echo "${run_id},${success_msg_grep},${cycles_grep},${failed_msg_grep},${syntax_error_msg_grep},${tag_bitflip_grep},${l1i_data_bitflip_grep},${false_l1i_hit_grep},${different_l1i_misses}" >>"$csv_file_path"
    fi
}

# Given a results.csv and a regexp filter, filter the runs that match the regexp
_filter_results_csv() {
    results_filepath=${1?No results.csv provided}
    # The regex pattern to use on the results to filter the runs. By default, only
    # selects runs with an injection that leads to a data bitflip: "[01],[01],[01],0,[01],1,[01],[01]"
    custom_pattern=${2:-[01],[01],[01],0,[01],1,[01],[01]}
    filtered_run_ids=($(gawk -v pat="^([a-f0-9]{32}),$custom_pattern" 'match($0, pat, a) {print a[1]}' <"$results_filepath"))
    echo "${filtered_run_ids[*]}"
}
