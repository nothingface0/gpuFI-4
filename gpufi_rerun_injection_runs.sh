#!/bin/bash

# gpuFI script for replaying all runs for a specific executable
# that have specific characteristics, e.g. those which include
# a tag bitflip that led to different cycles.
# After rerunning, a log in the format of <run_id>.log is created
# at the base directory of gpuFI and the results.csv file is updated
# based on the new log file.
#
# This script is mostly for debugging purposes, so that you can
# rerun specific interesting runs after changing something in the code base,
# without having to rerun every single run in results.csv.
#
# Requires CUDA_EXECUTABLE_PATH, CUDA_EXECUTABLE_ARGS and GPU_ID arguments,
# with the option of also specifying the CUSTOM_PATTERN.

source gpufi_utils.sh

# Complete command for CUDA executable
CUDA_EXECUTABLE_PATH=
CUDA_EXECUTABLE_ARGS=""
# Needed to locate the archive of the configs
GPU_ID=

# The regex pattern to use on the results to filter the runs. By default, only
# selects runs with an injection that leads to a data bitflip but NOT to a syntax error.
CUSTOM_PATTERN="[01],[01],[01],0,[01],1,[01],[01]"

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
find_injection_runs() {
    echo -n "Looking for runs which match $CUSTOM_PATTERN in $(_get_gpufi_analysis_path)/results/results.csv..."
    # Find the runs and add them into an array. Ignore those with syntax errors.
    FILTERED_RUN_IDS=($(_filter_results_csv "$(_get_gpufi_analysis_path)/results/results.csv") "$CUSTOM_PATTERN")
    export FILTERED_RUN_IDS
    echo "Done. Found ${#FILTERED_RUN_IDS[@]} runs."
}

# For each run id, run the gpufi_replay_run script.
run_simulator() {
    # Run each run_id serially for now
    for run_id in "${FILTERED_RUN_IDS[@]}"; do
        echo "Rerunning run $run_id"
        bash gpufi_replay_run.sh CUDA_EXECUTABLE_PATH="$CUDA_EXECUTABLE_PATH" CUDA_EXECUTABLE_ARGS="$CUDA_EXECUTABLE_ARGS" GPU_ID=$GPU_ID RUN_ID=$run_id
    done
}

# Look for the execution log in the same directory and update the results.csv file.
update_results() {
    # Get values from executable analysis file
    source "$(_get_gpufi_analysis_path)/executable_analysis.sh"
    for run_id in "${FILTERED_RUN_IDS[@]}"; do
        if [ ! -f "${run_id}.log" ]; then
            echo "${run_id}.log not found, skipping"
            continue
        fi
        echo "Examining run $run_id"
        _examine_log_file "${run_id}.log" "$_TOTAL_CYCLES" "$_L1I_CACHE_TOTAL_MISSES"
        _update_csv_file $run_id $success_msg_grep $cycles_grep $failed_msg_grep $syntax_error_msg_grep $tag_bitflip_grep $data_bitflip_grep $false_l1i_hit_grep $l1i_misses_grep
    done
}

### Script execution sequence ###
declare -a steps=(preliminary_checks
    find_injection_runs
    run_simulator
    update_results)

for step in "${steps[@]}"; do
    eval "do_${step}=1"
done

# Parse command line arguments -- use <key>=<value> to override any variable declared at the top of the script.
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
        echo "Replay step: $step"
        # Run the actual function
        eval "$step"
    else
        echo "Skipping step: $step"
    fi
done
