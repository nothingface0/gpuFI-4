#!/bin/bash

# gpuFI helper script for automatic recalculation of the results.csv
# entries, based on the execution logs of each run.
#
# Given the executable path, args and gpu_id, it finds results.csv,
# reads each run_id then tries to find the equivalent .log file in the
# current directory (or .gpufi_execution_logs). Then, it re-runs the
# _examine_log_file function, and updates the csv for each run.
# This is to clean up after messes you've made.

source gpufi_utils.sh

# Complete command for CUDA executable
CUDA_EXECUTABLE_PATH=
CUDA_EXECUTABLE_ARGS=""
# Needed to locate the archive of the configs
GPU_ID=

preliminary_checks() {
    if [ -z "$CUDA_EXECUTABLE_PATH" ]; then
        echo "Please provide a valid CUDA executable to run"
        exit 1
    fi

    if [ ! -f "$CUDA_EXECUTABLE_PATH" ]; then
        echo "File $CUDA_EXECUTABLE_PATH does not exist, please provide a valid executable"
        exit 1
    fi

    if ! _is_gpu_id_valid "$GPU_ID"; then
        echo "No valid GPU_ID was given, please provide a valid GPU_ID, e.g. SM7_QV100"
        exit 1
    fi

    if [ -z "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
        echo "GPGPU-Sim's setup_environment has not been run!"
        exit 1
    fi

    if [ ! -d "$(_get_gpufi_analysis_path)" ]; then
        echo "Could not find the executable's analysis path: $(_get_gpufi_analysis_path)"
        exit 1
    fi

    if [ ! -f "$(_get_gpufi_analysis_path)/results/results.csv" ]; then
        echo "results.csv not not found in $(_get_gpufi_analysis_path)/results"
        exit 1
    fi
    if [ ! -f "./gpufi_replay_run.sh" ]; then
        echo "This script must be executed from the root gpuFI-4 directory."
        exit 1
    fi
}

# Look into the results.csv file for runs where the run was successful, but
# cycles were different, without a tag or data bitflip taking place.
find_run_list() {
    echo -n "Looking for runs..."
    # Find the runs and add them into an array.
    BUGGY_RUN_IDS=($(gawk -v pat="^([a-f0-9]{32}),1,0,0,0,0,0,0,0$" 'match($0, pat, a) {print a[1]}' <"$(_get_gpufi_analysis_path)/results/results.csv"))
    export BUGGY_RUN_IDS
    echo "Done. Found ${#BUGGY_RUN_IDS[@]} runs."
}

update_results() {
    # Get values from executable analysis file
    source "$(_get_gpufi_analysis_path)/executable_analysis.sh"
    first_line=1
    while read -r line; do

        # Skip first line
        if [ $first_line -eq 1 ]; then
            first_line=0
            continue
        fi
        run_id=$(echo $line | cut -d',' -f1)
        if [ -f "./${run_id}.log" ]; then
            log_file="./${run_id}.log"
        elif [ -f "./.gpufi_execution_logs/${run_id}.log" ]; then
            log_file="./.gpufi_execution_logs/${run_id}.log"
        else
            continue
        fi
        _examine_log_file "$log_file" "$_TOTAL_CYCLES" "$_L1I_CACHE_TOTAL_MISSES"
        _update_csv_file $run_id $success_msg_grep $cycles_grep $failed_msg_grep $syntax_error_msg_grep $tag_bitflip_grep $data_bitflip_grep $false_l1i_hit_grep $l1i_misses_grep
    done <"$(_get_gpufi_analysis_path)/results/results.csv"
}

### Script execution sequence ###
declare -a steps=(preliminary_checks
    update_results)

for step in "${steps[@]}"; do
    eval "do_${step}=1"
done

# Parse command line arguments -- use <key>=<value> to override any variable declared above.
for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"
    eval "$KEY=\"$VALUE\""
done

# The actual replay procedure.
# For each step, check if the appropriate flag is enabled.
for step in "${steps[@]}"; do
    step_flag_name=do_$step
    if [ "${!step_flag_name}" -ne 0 ]; then
        echo "Step: $step"
        # Run the actual function
        eval "$step"
    else
        echo "Skipping step: $step"
    fi
done
