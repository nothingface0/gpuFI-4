#!/bin/bash

# gpuFI script for replaying a specific run, given the run_id,
# the CUDA_EXECUTABLE_PATH, the CUDA_EXECUTABLE_ARGS and the
# GPU_ID
#

source gpufi_utils.sh

# Complete command for CUDA executable
CUDA_EXECUTABLE_PATH=
CUDA_EXECUTABLE_ARGS=""
# Needed to locate the archive of the configs
GPU_ID=
RUN_ID=
_OUTPUT_LOG=

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

    if [ -z "$RUN_ID" ] || [ ${#RUN_ID} -ne 32 ]; then
        echo "${#RUN_ID}"
        echo "Please provide a valid md5sum for RUN_ID"
        exit 1
    fi

    if [ ! -f "$(_get_gpufi_analysis_path)/results/configs/$RUN_ID.tar.gz" ]; then
        echo "Run ID $RUN_ID not found"
        exit 1
    fi

    if [ -z "$_OUTPUT_LOG" ]; then
        _OUTPUT_LOG="./${RUN_ID}.log"
    fi

}

extract_config() {
    tmpdir=/tmp/gpufi_replay_config
    rm -rf $tmpdir # In case it exists, remove it
    mkdir -p $tmpdir
    tar -xf "$(_get_gpufi_analysis_path)/results/configs/$RUN_ID.tar.gz" --directory $tmpdir
    mv $tmpdir/* ./gpgpusim.config
    rm -rf $tmpdir
}

run_simulator() {
    source "$(_get_gpufi_analysis_path)/executable_analysis.sh"
    timeout $((_TIMEOUT_VALUE)) "$CUDA_EXECUTABLE_PATH" $CUDA_EXECUTABLE_ARGS >$_OUTPUT_LOG 2>&1
}

verify_run_id() {
    md5hash=$(_calculate_md5_hash "gpgpusim.config" "$CUDA_EXECUTABLE_PATH" "$(_sanitize_string $CUDA_EXECUTABLE_ARGS)")
    if [ "$md5hash" != "$RUN_ID" ]; then
        echo "Could not verify run id for run $RUN_ID"
        exit 1
    else
        echo "Successfully verified run $RUN_ID"
    fi
}

### Script execution sequence ###
declare -a steps=(preliminary_checks
    extract_config
    verify_run_id
    run_simulator)

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
        echo "Replay step: $step"
        # Run the actual function
        eval "$step"
    else
        echo "Skipping step: $step"
    fi
done
