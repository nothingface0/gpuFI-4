#!/bin/bash

# gpuFI helper script to analyze an executable before an injection campaign.
# It takes in a combination of gpgpusim.config corresponding to a GPU, a CUDA executable
# (must have SASS < sm_19 embedded) and its arguments
# and runs it in order to analyse the resources it uses per kernel, in order
# to configure the gpuFI campaigns accordingly.
#
# The script will create a .gpufi directory in the same directory the executable is in,
# with a <GPU_ID>/<CUDA_EXECUTABLE_ARGS sanitized> subdirectory, where the analysis output
# will be stored.
#
# Called as following:
# bash analyze_executable.sh CUDA_EXECUTABLE_PATH=<path to executable> \
#                           CUDA_EXECUTABLE_ARGS="args in double quotes please" \
#                           GPGPU_SIM_CONFIG_PATH=<path to gpgpusim.config> \
#                           GPU_ID=<id/name of the GPU the gpgpusim.config corresponds to>

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

_get_gpufi_analysis_path() {
    echo "$(dirname $CUDA_EXECUTABLE_PATH)/.gpufi/$GPU_ID/$(_sanitize $CUDA_EXECUTABLE_ARGS)"
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
    if [ -f "$(_get_gpufi_analysis_path)/.analysis_complete" ]; then
        echo "$CUDA_EXECUTABLE_PATH with args $CUDA_EXECUTABLE_ARGS has already been analyzed, skipping."
        exit 0
    fi
}

# Create the .gpufi directory where the necessary configs will be stored
create_directories() {
    dirname="$(_get_gpufi_analysis_path)"
    echo "Creating directory $dirname"
    mkdir -p "$dirname"

}

_create_kernel_directories() {
    # Create subdirs for each kernel
    for kernel_name in $KERNEL_NAMES; do
        kernel_dirname="$(_get_gpufi_analysis_path)/$kernel_name"
        echo "Creating kernel subdirectory: $kernel_dirname"
        mkdir -p $kernel_dirname
    done
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
    if [ $gpufi_profile -ne 1 ]; then
        echo "gpufi_profile must either be 0 or 3 for running analysis"
        exit 1
    fi
}

execute_executable() {
    echo "TODO: execute cuda executable, store output to file, count time."
    # Copy the config file just in case
    cp "$GPGPU_SIM_CONFIG_PATH" "$(_get_gpufi_analysis_path)/$(basename $GPGPU_SIM_CONFIG_PATH)"

    # Time the execution
    SECONDS=0

    # Run the executable, store the output
    eval "$CUDA_EXECUTABLE_PATH $CUDA_EXECUTABLE_ARGS" >"$(_get_gpufi_analysis_path)/out.log"

    export CUDA_EXECUTABLE_EXECUTION_TIME=$SECONDS
    echo "Execution took $CUDA_EXECUTABLE_EXECUTION_TIME seconds"

}

# Stuff to do after execution is complete
post_execution_actions() {
    _parse_executable_output
    _create_kernel_directories # Depends on parsing the execution output first
    _create_per_kernel_config
}

_parse_executable_output() {
    echo "TODO: parse the output of the executable and extract per-kernel information"
    output_log="$(_get_gpufi_analysis_path)/out.log"
    if [ ! -f "$output_log" ]; then
        echo "Could not find the execution log: $output_log"
        exit 1
    fi

    export TOTAL_CYCLES
    TOTAL_CYCLES=$(grep "gpu_tot_sim_cycle" "$output_log" | tail -1 | gawk -v pat="gpu_tot_sim_cycle = ([0-9]+)" 'match($0, pat, a) {print a[1]}')
    export TIMEOUT_VALUE
    TIMEOUT_VALUE=$((CUDA_EXECUTABLE_EXECUTION_TIME * 2))
    regex_mangled_name="(_Z[0-9_[:alnum:]]+)"
    export KERNEL_NAMES
    KERNEL_NAMES=$(grep -E "kernel_name = $regex_mangled_name" "$output_log" | uniq | gawk -v pat="kernel_name = $regex_mangled_name" 'match($0, pat, a) {print a[1]}')

    # We need:
    # - Per GPU config+executable+args: cycles total, timeout
    # - Per GPU config+executable+args+kernel: name, smem, lmem, max registers, shaders, active cycles
    for kernel_name in $KERNEL_NAMES; do
        echo -n "Analyzing kernel $kernel_name..."
        regex_kernel_regs_mem="Kernel '$kernel_name' : regs=([0-9]+), lmem=([0-9]+), smem=([0-9]+), cmem=([0-9]+)"
        regs_mems=$(cat "$output_log" | gawk -v pat="$regex_kernel_regs_mem" 'match($0, pat, a) {print a[1], a[2], a[3], a[4]}')
        regs_mems=(${regs_mems// / })

        var_name_kernel_lmem="KERNEL_${kernel_name}_LMEM_USED"
        eval "export $var_name_kernel_lmem=${regs_mems[1]}"
        var_name_kernel_smem="KERNEL_${kernel_name}_SMEM_USED"
        eval "export $var_name_kernel_smem=${regs_mems[2]}"
        var_name_kernel_cmem="KERNEL_${kernel_name}_CMEM_USED"
        eval "export $var_name_kernel_cmem=${regs_mems[3]}"
        echo "Done"

        regex_kernel_max_active_regs="gpuFI: Kernel = $kernel_name, max active regs = ([0-9]+)"
        max_active_regs=$(cat "$output_log" | gawk -v pat="$regex_kernel_max_active_regs" 'match($0, pat, a) {print a[1]}')
        var_name_kernel_regs="KERNEL_${kernel_name}_MAX_ACTIVE_REGS"
        eval "export $var_name_kernel_regs=$max_active_regs"

        regex_kernel_used_shaders="gpuFI: Kernel = $kernel_name used shaders: ([0-9[:space:]]+)"
        shaders_used=$(cat "$output_log" | gawk -v pat="$regex_kernel_used_shaders" 'match($0, pat, a) {print a[1]}')
        shaders_used=(${shaders_used// / })
        var_name_kernel_shaders="KERNEL_${kernel_name}_SHADERS_USED"
        eval "export $var_name_kernel_shaders=\"${shaders_used[*]}\""
        #TODO: active cycles
    done
    #touch "$(_get_gpufi_analysis_path)/.analysis_complete"
}

_create_per_kernel_config() {
    for kernel_name in $KERNEL_NAMES; do
        var_name_kernel_regs="KERNEL_${kernel_name}_REGS_USED"
        var_name_kernel_lmem="KERNEL_${kernel_name}_LMEM_USED"
        var_name_kernel_smem="KERNEL_${kernel_name}_SMEM_USED"
        # Unused for now
        # var_name_kernel_cmem="KERNEL_${kernel_name}_CMEM_USED"
    done
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
    check_gpufi_profile
    create_directories
    execute_executable
    post_execution_actions
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

# The actual analysis procedure.
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
