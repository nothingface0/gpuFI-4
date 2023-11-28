#!/bin/bash

# gpuFI helper script to analyze an executable before an injection campaign.
# It takes in a combination of gpgpusim.config corresponding to a GPU, a CUDA executable
# (must have SASS < sm_19 embedded) and its arguments
# and runs it in order to analyse the resources it uses per kernel, in order
# to configure the gpuFI campaigns accordingly.

set -ex

# The full path of the executable to analyze
CUDA_EXECUTABLE_PATH=

# The arguments that the executable needs in order to run
CUDA_EXECUTABLE_ARGS=

# The full path to the gpgpusim.config file to use
GPGPU_SIM_CONFIG_PATH=

# The ID of the GPU the config corresponds to, e.g. SM7_QV100
GPU_ID=

# Function to sanitize args to a folder name
# From here: https://stackoverflow.com/a/44811468/6562491
_sanitize() {
    local s="${*?need a string}" # receive input in first argument
    s="${s//[^[:alnum:]]/-}"     # replace all non-alnum characters to -
    s="${s//+(-)/-}"             # convert multiple - to single -
    s="${s/#-/}"                 # remove - from start
    s="${s/%-/}"                 # remove - from end
    echo "${s,,}"                # convert to lowercase
}

preliminary_checks() {
    # Check script args, check if analysis has been already run.
    if [ ! -f "$CUDA_EXECUTABLE_PATH" ]; then
        echo "File $CUDA_EXECUTABLE_PATH does not exist, please provide a valid executable"
        exit 1
    fi

    if [ ! -f "$GPGPU_SIM_CONFIG_PATH" ]; then
        echo "File $GPGPU_SIM_CONFIG_PATH does not exist, please provide a valid gpgpusim.config"
        exit 1
    fi

    if [ -z "$GPU_ID" ]; then
        echo "No GPU id was given, please provide a valid GPU id, e.g. SM7_QV100"
        exit 1
    fi

    if [ -z "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
        echo "GPGPU-Sim's setup_environment has not been run!"
        exit 1
    fi

    # Try to see if analysis has already been run for this specfic combination of
    # GPU id, executable and args.
    if [ -d "$(dirname $CUDA_EXECUTABLE_PATH)/.gpufi/$GPU_ID/$(_sanitize $CUDA_EXECUTABLE_ARGS)" ]; then
        echo "$CUDA_EXECUTABLE_PATH with args $CUDA_EXECUTABLE_ARGS has already been analyzed, skipping."
        exit 0
    fi
}

create_directories() {
    dirname="$(dirname "$CUDA_EXECUTABLE_PATH")/.gpufi/$GPU_ID/$(_sanitize "$CUDA_EXECUTABLE_ARGS")"
    echo "Creating directory $dirname"
    mkdir -p "$dirname"
}

check_gpufi_profile() {
    echo "TODO: Make sure that gpufi_profile is set to 1 or 3"
    gpufi_profile_regex="^[[:space:]]*-gpufi_profile[[:space:]]+([0-9])"
    if ! grep -E $gpufi_profile_regex "$GPGPU_SIM_CONFIG_PATH" --only-matching; then
        echo "No gpufi_profile parameter found in $GPGPU_SIM_CONFIG_PATH"
        exit 1
    fi

    # Get the gpufi_profile value from the config
    gpufi_profile=$(cat $GPGPU_SIM_CONFIG_PATH | gawk -v pat=$gpufi_profile_regex 'match($0, pat, a) {print a[1]}')
    if [ $gpufi_profile -ne 1 ] && [ $gpufi_profile -ne 3 ]; then
        echo "gpufi_profile must either be 0 or 3 for running analysis"
        exit 1
    fi
}

execute_executable() {
    echo "TODO: execute cuda executable, store output to file, count time."
}

parse_executable_output() {
    echo "TODO: parse the output of the executable and extract per-kernel information"
    # We need:
    # - Per GPU config+executable+args: cycles total, timeout
    # - Per GPU config+executable+args+kernel: name, smem, lmem, max registers, shaders, active cycles
}

create_gpufi_configs() {
    echo "TODO: create directories and files per GPU/executable/arguments/kernel combination"
    _create_cycles_txt
}

_create_cycles_txt() {
    echo "TODO: create cycles.in --> cycles.txt"
}

### Main script
declare -a analysis_steps=(
    preliminary_checks
    create_directories
    check_gpufi_profile
    execute_executable
    parse_executable_output
    create_gpufi_configs
)

# Create dynamic flags to selectively disable/enable steps of the analysis if needed.
# Those flags are named "do_" with the name of the function.
# We set those flags to 1 by default.
for step in "${analysis_steps[@]}"; do
    eval "do_${step}=1"
done

# Parse command line arguments -- use <key>=<value> to override the flags mentioned above.
for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"
    eval "$KEY=\"$VALUE\""
done

# The actual installation procedure.
# For each step, check if the appropriate flag is enabled.
for step in "${analysis_steps[@]}"; do

    analysis_step_flag_name=do_$step
    if [ "${!analysis_step_flag_name}" -ne 0 ]; then
        echo "Analysis step: $step"
        # Run the actual function
        eval "$step"
    else
        echo "Skipping step: $step"
    fi
done
