#!/bin/bash

# gpuFI helper script to analyze an executable before an injection campaign.
# It takes in a combination of a CUDA executable
# (must have SASS < sm_19 embedded), its arguments and a GPU id (e.g. SM7_QV100)
# and runs it in order to analyse the resources it uses per kernels and
# configure the gpuFI campaigns accordingly.
#
# Analysis consists of running the executable twice: once with gpufi_profile=3,
# to get a baseline execution (total cycles and a timeout value) and another
# with gpufi_profile=1 (to get the used resources per kernel).
#
# The script will create a .gpufi directory in the same directory the executable is in,
# with a <GPU_ID>/<CUDA_EXECUTABLE_ARGS sanitized> subdirectory, where the analysis output
# will be stored.
#
# Called as following:
# bash analyze_executable.sh CUDA_EXECUTABLE_PATH=<path to executable> \
#                           CUDA_EXECUTABLE_ARGS="args in double quotes please" \
#                           GPU_ID=<id/name of the GPU the gpgpusim.config corresponds to, e.g. SM7_QV100>

set -e
source gpufi_utils.sh

# The full path of the executable to analyze
CUDA_EXECUTABLE_PATH=

# The arguments that the executable needs in order to run
CUDA_EXECUTABLE_ARGS=

# The ID of the GPU the config corresponds to, e.g. SM7_QV100
GPU_ID=

# The full path to the gpgpusim.config file to use. Will be detected automatically,
# using the GPU_ID, but can also be overriden from the script arguments.
_GPGPU_SIM_CONFIG_PATH=

# Checks to do before running the executable analysis
preliminary_checks() {
    # Check script args, check if analysis has been already run.
    if [ ! -f "$CUDA_EXECUTABLE_PATH" ]; then
        echo "File $CUDA_EXECUTABLE_PATH does not exist, please provide a valid executable"
        exit 1
    fi

    if [ -n "$_GPGPU_SIM_CONFIG_PATH" ] && [ ! -f "$_GPGPU_SIM_CONFIG_PATH" ]; then
        # A path to the config has been supplied, but the file does not exist
        echo "File $_GPGPU_SIM_CONFIG_PATH does not exist, please provide a valid gpgpusim.config"
        exit 1
    fi

    if ! grep -q "gpufi_profile" "$_GPGPU_SIM_CONFIG_PATH"; then
        echo "$_GPGPU_SIM_CONFIG_PATH does not have gpufi configuration!"
    fi
    if ! _is_gpu_id_valid $GPU_ID; then
        echo "No valid GPU_ID was given, please provide a valid GPU id, e.g. SM7_QV100"
        exit 1
    fi

    if [ -z "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
        echo "GPGPU-Sim's setup_environment has not been run!"
        exit 1
    fi

    # Try to see if analysis has already been run for this specfic combination of
    # GPU id, executable and args.
    if [ -f "$(_get_gpufi_analysis_path)/.analysis_complete" ]; then
        echo "\"$CUDA_EXECUTABLE_PATH\" with args \"$CUDA_EXECUTABLE_ARGS\" has already been analyzed, skipping."
        exit 0
    fi
}

# Create the .gpufi directory where the necessary configs will be stored
create_directories() {
    dirname="$(_get_gpufi_analysis_path)"
    echo "Creating directory $dirname"
    mkdir -p "$dirname"
}

# Create per-kernel subdirectories under the main analysis
# directory to store per-kernel information.
_create_kernel_directories() {
    # Create subdirs for each kernel
    for kernel_name in $KERNEL_NAMES; do
        kernel_dirname="$(_get_gpufi_analysis_path)/$kernel_name"
        echo "Creating kernel subdirectory: $kernel_dirname"
        mkdir -p $kernel_dirname
    done
}

check_gpufi_profile() {
    target_value=$1
    # Make sure that gpufi_profile is found in the config file
    gpufi_profile_regex="^[[:space:]]*-gpufi_profile[[:space:]]+([0-9])"
    if ! grep -E $gpufi_profile_regex "$_GPGPU_SIM_CONFIG_PATH" --only-matching >/dev/null; then
        echo "No gpufi_profile parameter found in $_GPGPU_SIM_CONFIG_PATH"
        exit 1
    fi

    # Get the gpufi_profile value from the config
    gpufi_profile=$(cat $_GPGPU_SIM_CONFIG_PATH | gawk -v pat=$gpufi_profile_regex 'match($0, pat, a) {print a[1]}')
    if [ $gpufi_profile -ne $target_value ]; then
        echo "gpufi_profile must be $target_value for analysing the executable"
        exit 1
    fi
}

# Change the gpufi_profile parameter in the _GPGPU_SIM_CONFIG_PATH file
change_gpufi_profile_normal() {
    sed -i -e "s/^-gpufi_profile.*$/-gpufi_profile 3/" ${_GPGPU_SIM_CONFIG_PATH}
}
change_gpufi_profile_profiling() {
    sed -i -e "s/^-gpufi_profile.*$/-gpufi_profile 1/" ${_GPGPU_SIM_CONFIG_PATH}
}

# Execute cuda executable, store output to file, count time.
execute_executable() {
    # Copy the config file just in case
    cp "$_GPGPU_SIM_CONFIG_PATH" "$(dirname $(_get_gpufi_analysis_path))/$(basename $_GPGPU_SIM_CONFIG_PATH)"

    # Time the execution
    SECONDS=0

    # Run the executable, store the output
    eval "$CUDA_EXECUTABLE_PATH $CUDA_EXECUTABLE_ARGS" >"$(_get_gpufi_analysis_path)/out.log"

    export CUDA_EXECUTABLE_EXECUTION_TIME=$SECONDS
    echo "Execution took $CUDA_EXECUTABLE_EXECUTION_TIME seconds"

}

# Parse the output of the executable and extract per-kernel information:
# - Kernel names,
# - Cycles that each kernel is active for,
# - Shaders used,
# - Max registers used,
# - LMEM size bits
# - SMEM size bits
# Exports the necessary variables to be used subsequently.
parse_executable_output() {
    output_log="$(_get_gpufi_analysis_path)/out.log"
    if [ ! -f "$output_log" ]; then
        echo "Could not find the execution log: $output_log"
        exit 1
    fi

    export TOTAL_CYCLES
    TOTAL_CYCLES=$(grep "gpu_tot_sim_cycle" "$output_log" | tail -1 | gawk -v pat="gpu_tot_sim_cycle = ([0-9]+)" 'match($0, pat, a) {print a[1]}')
    export TIMEOUT_VALUE
    TIMEOUT_VALUE=$((CUDA_EXECUTABLE_EXECUTION_TIME))
    regex_mangled_name="(_Z[0-9_[:alnum:]]+)"
    export KERNEL_NAMES
    KERNEL_NAMES=$(grep -E "kernel_name = $regex_mangled_name" "$output_log" | uniq | gawk -v pat="kernel_name = $regex_mangled_name" 'match($0, pat, a) {print a[1]}')
    export L1I_CACHE_TOTAL_MISSES
    L1I_CACHE_TOTAL_MISSES=$(grep "L1I_total_cache_misses" "$output_log" | tail -1 | gawk -v pat="L1I_total_cache_misses = ([0-9]+)" 'match($0, pat, a) {print a[1]}')

    # We need:
    # - Per GPU config+executable+args: cycles total, timeout
    # - Per GPU config+executable+args+kernel: name, smem, lmem, max registers, shaders, active cycles
    for kernel_name in $KERNEL_NAMES; do
        echo -n "Analyzing kernel $kernel_name..."
        regex_kernel_regs_mem="Kernel '$kernel_name' : regs=([0-9]+), lmem=([0-9]+), smem=([0-9]+), cmem=([0-9]+)"
        regs_mems=$(cat "$output_log" | gawk -v pat="$regex_kernel_regs_mem" 'match($0, pat, a) {print a[1], a[2], a[3], a[4]}')
        regs_mems=(${regs_mems// / })

        # LMEM, SMEM, CMEM used
        var_name_kernel_lmem="KERNEL_${kernel_name}_LMEM_USED_BITS"
        eval "export $var_name_kernel_lmem=${regs_mems[1]}"
        var_name_kernel_smem="KERNEL_${kernel_name}_SMEM_USED_BITS"
        eval "export $var_name_kernel_smem=${regs_mems[2]}"
        var_name_kernel_cmem="KERNEL_${kernel_name}_CMEM_USED_BITS"
        eval "export $var_name_kernel_cmem=${regs_mems[3]}"

        # Max registers used
        regex_kernel_max_active_regs="gpuFI: Kernel = $kernel_name, max active regs = ([0-9]+)"
        max_active_regs=$(cat "$output_log" | gawk -v pat="$regex_kernel_max_active_regs" 'match($0, pat, a) {print a[1]}')
        var_name_kernel_regs="KERNEL_${kernel_name}_MAX_REGISTERS_USED"
        eval "export $var_name_kernel_regs=$max_active_regs"

        # Shaders used
        regex_kernel_used_shaders="gpuFI: Kernel = $kernel_name used shaders: ([0-9[:space:]]+)"
        shaders_used=$(cat "$output_log" | gawk -v pat="$regex_kernel_used_shaders" 'match($0, pat, a) {print a[1]}')
        shaders_used=(${shaders_used// / })
        var_name_kernel_shaders="KERNEL_${kernel_name}_SHADERS_USED"
        eval "export $var_name_kernel_shaders=\"${shaders_used[*]}\""

        # Start-stop cycles that the kernel is active for.
        regex_kernel_active_cycles="gpuFI: Kernel = ([0-9]+) with name = $kernel_name, started on cycle = ([0-9]+) and finished on cycle = ([0-9]+)"
        kernel_active_cycles_sets=$(cat "$output_log" | gawk -v pat="$regex_kernel_active_cycles" 'match($0, pat, a) {print a[1], a[2], a[3]}')
        # A single kernel may be launched many times
        OLD_IFS=$IFS
        IFS=$'\n' # Temporarily split with newlines to separate the multi-line output of gawk
        kernel_active_cycles=""
        for set in $kernel_active_cycles_sets; do
            IFS=' '           # Start-stop cycles are seperated with a space
            set=(${set// / }) # Split into an array
            # Sanity check
            # echo "Kernel $kernel_name launch# ${set[0]}, started ${set[1]}, stopped ${set[2]}"
            kernel_active_cycles="${kernel_active_cycles},${set[1]}:${set[2]}"
            IFS=$'\n' # Split with newlines again
        done
        IFS=$OLD_IFS
        var_name_kernel_active_cycles="KERNEL_${kernel_name}_ACTIVE_CYCLES"
        eval "export $var_name_kernel_active_cycles=\"${kernel_active_cycles[*]}\""
        echo "Done"
    done
}

# Create an analysis file for the specific executable: cycles total, timeout expected
# This gets called after execution with gpufi_profile=3, i.e. a clean execution,
# so the TIMEOUT_VALUE might be a bit underestimated, which is why we multiply it
# by some factor to allow for some fluctuation in execution time.
create_per_executable_analysis_file() {
    rm -rf "$(_get_gpufi_analysis_path)/executable_analysis.sh"
    {
        echo "_TOTAL_CYCLES=${TOTAL_CYCLES}"
        echo "_TIMEOUT_VALUE=$((TIMEOUT_VALUE * 5))"
        echo "_L1I_CACHE_TOTAL_MISSES=${L1I_CACHE_TOTAL_MISSES}"
    } >>"$(_get_gpufi_analysis_path)/executable_analysis.sh"
}

# For each kernel inside the executable, create a subdirectory where
# analysis information will be stored
_create_per_kernel_analysis_file() {
    _create_kernel_directories # Create per-kernel subdirs, depends on parsing the execution output first

    # List of all shaders used by all kernels
    merged_kernel_shaders_used=""
    # gpuFI TODO: Does it make sense to make an aggregate of max registers of all kernels? If yes, how?
    # merged_kernel_max_registers=0
    # gpuFI TODO: Does it make sense to make an aggregate of LMEM, SMEM, CMEM for all kernels?

    for kernel_name in $KERNEL_NAMES; do
        per_kernel_analysis_file_path="$(_get_gpufi_analysis_path)/$kernel_name/kernel_analysis.sh"
        rm -rf "$per_kernel_analysis_file_path"

        var_name_kernel_shaders="KERNEL_${kernel_name}_SHADERS_USED"
        var_name_kernel_regs="KERNEL_${kernel_name}_MAX_REGISTERS_USED"
        var_name_kernel_lmem="KERNEL_${kernel_name}_LMEM_USED_BITS"
        var_name_kernel_smem="KERNEL_${kernel_name}_SMEM_USED_BITS"
        var_name_kernel_cmem="KERNEL_${kernel_name}_CMEM_USED_BITS" # Unused for now

        # Append shaders used in this kernel
        merged_kernel_shaders_used="$merged_kernel_shaders_used ${!var_name_kernel_shaders}"

        {
            echo "_SHADERS_USED=\"${!var_name_kernel_shaders}\""
            echo "_MAX_REGISTERS_USED=${!var_name_kernel_regs}"
            # If LMEM, SMEM or CMEM are 0, a random positive int is used.
            tmp=${!var_name_kernel_lmem}
            [ $tmp -eq 0 ] && tmp=999999 || tmp=$((tmp * 8))
            echo "_LMEM_SIZE_BITS=$tmp"
            tmp=${!var_name_kernel_smem}
            [ $tmp -eq 0 ] && tmp=999999 || tmp=$((tmp * 8))
            echo "_SMEM_SIZE_BITS=$tmp"
            tmp=${!var_name_kernel_cmem}
            [ $tmp -eq 0 ] && tmp=999999 || tmp=$((tmp * 8))
            echo "_CMEM_SIZE_BITS=$tmp"
        } >>"$per_kernel_analysis_file_path"
    done

    # Output all shaders used by all kernels into a single file too.
    merged_kernel_shaders_used=("${merged_kernel_shaders_used// / }")
    # Find unique values in shaders used.
    merged_kernel_shaders_used=$(
        for shader in $merged_kernel_shaders_used; do
            echo $shader
        done | sort -g | uniq
    )
    # Concatenate into a single string
    merged_kernel_shaders_used=$(
        tmp=""
        for shader in $merged_kernel_shaders_used; do
            tmp="$tmp $shader"
        done
        echo $tmp
    )

    merged_kernel_analysis_file_path="$(_get_gpufi_analysis_path)/merged_kernel_analysis.sh"
    rm -rf "$merged_kernel_analysis_file_path"
    {
        echo "_SHADERS_USED=\"${merged_kernel_shaders_used}\""
        echo "_MAX_REGISTERS_USED=1" # gpuFI TODO
        echo "_LMEM_SIZE_BITS=1"     # gpuFI TODO
        echo "_SMEM_SIZE_BITS=1"     # gpuFI TODO
        echo "_CMEM_SIZE_BITS=1"     # gpuFI TODO
    } >>"$merged_kernel_analysis_file_path"
}

# Create directories and files per GPU/executable/arguments/kernel combination
create_per_kernel_configs() {
    _create_per_kernel_analysis_file
    _create_per_kernel_cycles_txt
}

# Getting the start-stop cycles of each kernel, create the files that
# contain every cycle that each kernel is active for.
_create_per_kernel_cycles_txt() {
    OLD_IFS=$IFS

    # File with the cycles of all the kernels
    merged_cycles_txt_file="$(_get_gpufi_analysis_path)/merged_cycles.txt"
    rm -rf "$merged_cycles_txt_file"

    # For each kernel executed, create a separate file
    for kernel_name in $KERNEL_NAMES; do
        cycles_txt_file="$(_get_gpufi_analysis_path)/$kernel_name/cycles.txt"
        if [ -f "$cycles_txt_file" ]; then
            rm -f "$cycles_txt_file"
        fi
        var_name_kernel_active_cycles="KERNEL_${kernel_name}_ACTIVE_CYCLES"
        IFS="," # Multiple invocations of the same kernel is split with comma
        for start_stop_cycle in ${!var_name_kernel_active_cycles}; do
            if [ -z "$start_stop_cycle" ]; then
                continue
            fi
            # Each kernel invocation is in the form of <startcycle>:<stopcycle>
            start=$(echo $start_stop_cycle | cut -d':' -f1)
            stop=$(echo $start_stop_cycle | cut -d':' -f2)
            # Output the sequence to file
            seq $start $stop >>$cycles_txt_file
        done
        IFS=","
        # Also append the cycles to the merged cycles file
        cat $cycles_txt_file >>$merged_cycles_txt_file
    done

    IFS=$OLD_IFS
}

# Modify the config's last cycle. Needed (with gpufi_profile=1) to print the kernel stats at the right time.
modify_total_cycles() {
    sed -i -e "s/^-gpufi_last_cycle.*$/-gpufi_last_cycle ${TOTAL_CYCLES}/" ${_GPGPU_SIM_CONFIG_PATH}
}

# Create an empty file which signals that analysis has run.
finalize_analysis() {
    touch "$(_get_gpufi_analysis_path)/.analysis_complete"
}

prepare_files() {
    _copy_gpgpusim_config $GPU_ID
}

### Main script
declare -a analysis_steps=(
    preliminary_checks
    prepare_files
    create_directories
    change_gpufi_profile_normal
    execute_executable
    parse_executable_output
    create_per_executable_analysis_file
    modify_total_cycles
    change_gpufi_profile_profiling
    execute_executable
    parse_executable_output
    create_per_kernel_configs
    finalize_analysis
)

# Create dynamic flags to selectively disable/enable steps of the analysis if needed.
# Those flags are named "do_" with the name of the function.
# We set those flags to 1 by dstepefault.
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
