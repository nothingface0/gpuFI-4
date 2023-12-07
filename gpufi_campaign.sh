#!/bin/bash

# gpuFI script for running an injection campaign for a combination of:
# - A specific gpgpusim.config (i.e. a specific GPU),
# - A specific CUDA executable and its arguments, and
# - A kernel of the executable to be targeted.
#
# Parameters meant for internal use start with "_".

# set -x
source gpufi_utils.sh
# Paths to files created from gpufi_analyze_executable.sh
_EXECUTABLE_ANALYSIS_FILE=
_KERNEL_ANALYSIS_FILE=
GPU_ID=
KERNEL_NAME=
# ---------------------------------------------- START ONE-TIME PARAMETERS ----------------------------------------------
# needed by gpgpu-sim for real register usage on PTXPlus mode
#export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.0

GPGPU_SIM_CONFIG_PATH=
TMP_DIR=./logs
CACHE_LOGS_DIR=./cache_logs
TMP_FILE=tmp.out
NUM_RUNS=1                             # How many runs to simulate of the given executable. Randomize injections on each run.
DELETE_LOGS=0                          # if 1 then all logs will be deleted at the end of the script
_NUM_AVAILABLE_CORES=$(($(nproc) - 1)) # How many instances of the simulator to run in parallel
# ---------------------------------------------- END ONE-TIME PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER GPGPU CARD PARAMETERS ----------------------------------------------

# ---------------------------------------------- END PER GPGPU CARD PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER KERNEL/APPLICATION PARAMETERS (+_GPUFI_PROFILE=1) ----------------------------------------------
# gpuFI TODO: Configuration in this section seems to have been intended for targeting a
# specific kernel of a speicific executable. But this does not make sense, as there is an option,
# KERNEL_INDICES, which allows gpuFI to target ALL the kernels. How will _MAX_REGISTERS_USED make
# sense for all the kernels then?

# Register size
# gpuFI TODO: Should this be hardcoded?
_DATATYPE_SIZE=32

# Complete command for CUDA executable
CUDA_EXECUTABLE_PATH=
CUDA_EXECUTABLE_ARGS=""

# total cycles for all kernels
_TOTAL_CYCLES=
# Time to wait for executable to finish running.
_TIMEOUT_VALUE=

# Per kernel config
# cycles.txt file for the kernel. Used for selecting a random cycle to perform the injection while
# the targeted kernel is running both in this script, and inside the simulator, when _GPUFI_PROFILE=2.
_CYCLES_FILE=
_MAX_REGISTERS_USED=
_SHADERS_USED=
_SUCCESS_MSG='Test PASSED'
_FAILED_MSG='Test FAILED'

# lmem and smem values are taken from gpgpu-sim ptx output per kernel
# e.g. GPGPU-Sim PTX: Kernel '_Z9vectorAddPKdS0_Pdi' : regs=8, lmem=0, smem=0, cmem=380
# if 0 put a random value > 0
_LMEM_SIZE_BITS=
_SMEM_SIZE_BITS=
# ---------------------------------------------- END PER KERNEL/APPLICATION PARAMETERS (+_GPUFI_PROFILE=1) ------------------------------------------------

_FAULT_INJECTION_OCCURRED="gpuFI: Fault injection"
_CYCLES_MSG="gpu_tot_sim_cycle ="

# Campaign runtime variables
_errors_masked=0
_errors_performance=0
_errors_sdc=0 # Silent data corruption
_errors_due=0 # Detected unrecoverable error (crash)

# ---------------------------------------------- START PER INJECTION CAMPAIGN PARAMETERS (_GPUFI_PROFILE=0) ----------------------------------------------
# gpuFI profile to run. Possible values:
# 0: perform injection campaign
# 1: get cycles of each kernel
# 2: get mean value of active threads, during all cycles in _CYCLES_FILE, per SM,
# 3: single fault-free execution
_GPUFI_PROFILE=0

# Which components to apply a bif flip to. Multiple ones can be
# specified with a colon, e.g. COMPONENTS_TO_FLIP=0:1 for both RF and local_mem). Possible values:
# 0: RF
# 1: local_mem
# 2: shared_mem
# 3: L1D_cache
# 4: L1C_cache
# 5: L1T_cache
# 6: L2_cache
# 7: L1I_cache
COMPONENTS_TO_FLIP=7

# 0: per thread bit flip, 1: per warp bit flip
PER_WARP=0

# in which kernels to inject the fault. e.g. 0: for all running kernels, 1: for kernel 1, 1:2 for kernel 1 & 2
KERNEL_INDICES=0

# in how many blocks (smems) to inject the bit flip
BLOCKS_NUM=1

# Function which initializes variables based on user config and edits the gpgpusim.config file.
initialize_config() {
    # Random number for choosing a random thread after gpufi_thread_rand % #threads operation in gpgpu-sim
    # The number 6000 looks hardcoded but it does not have a significance. It should just
    # be >= the total number of threads, as it will be modulo'd with the total number during
    # execution.
    gpufi_thread_rand=$(shuf -i 0-6000 -n 1)
    # random number for choosing a random warp after gpufi_warp_rand % #warp operation in gpgpu-sim
    gpufi_warp_rand=$(shuf -i 0-6000 -n 1)
    gpufi_total_cycle_rand=-1
    if [[ "$_GPUFI_PROFILE" -eq 0 ]]; then
        # random cycle for fault injection
        gpufi_total_cycle_rand="$(shuf ${_CYCLES_FILE} -n 1)"
    fi
    # in which registers to inject the bit flip
    gpufi_register_rand_n="$(shuf -i 1-${_MAX_REGISTERS_USED} -n 1)"
    gpufi_register_rand_n="${gpufi_register_rand_n//$'\n'/:}"
    # example: if -i 1-32 -n 2 then the two commands below will create a value with 2 random numbers, between [1,32] like 3:21. Meaning it will flip 3 and 21 bits.
    gpufi_reg_bitflip_rand_n="$(shuf -i 1-${_DATATYPE_SIZE} -n 1)"
    gpufi_reg_bitflip_rand_n="${gpufi_reg_bitflip_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for local memory bit flips
    gpufi_local_mem_bitflip_rand_n="$(shuf -i 1-${_LMEM_SIZE_BITS} -n 3)"
    gpufi_local_mem_bitflip_rand_n="${gpufi_local_mem_bitflip_rand_n//$'\n'/:}"
    # random number for choosing a random block after gpufi_block_rand % #smems operation in gpgpu-sim
    gpufi_block_rand=$(shuf -i 0-6000 -n 1)
    # same format like gpufi_reg_bitflip_rand_n but for shared memory bit flips
    gpufi_shared_mem_bitflip_rand_n="$(shuf -i 1-${_SMEM_SIZE_BITS} -n 1)"
    gpufi_shared_mem_bitflip_rand_n="${gpufi_shared_mem_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 data cache fault injections
    gpufi_l1d_shader_rand_n="$(shuf -e ${_SHADERS_USED} -n 1)"
    gpufi_l1d_shader_rand_n="${gpufi_l1d_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 data cache bit flips
    gpufi_l1d_cache_bitflip_rand_n="$(shuf -i 1-${L1D_SIZE_BITS} -n 1)"
    gpufi_l1d_cache_bitflip_rand_n="${gpufi_l1d_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 constant cache fault injections
    gpufi_l1c_shader_rand_n="$(shuf -e ${_SHADERS_USED} -n 1)"
    gpufi_l1c_shader_rand_n="${gpufi_l1c_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 constant cache bit flips
    gpufi_l1c_cache_bitflip_rand_n="$(shuf -i 1-${L1C_SIZE_BITS} -n 1)"
    gpufi_l1c_cache_bitflip_rand_n="${gpufi_l1c_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 texture cache fault injections
    gpufi_l1t_shader_rand_n="$(shuf -e ${_SHADERS_USED} -n 1)"
    gpufi_l1t_shader_rand_n="${gpufi_l1t_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 texture cache bit flips
    gpufi_l1t_cache_bitflip_rand_n="$(shuf -i 1-${L1T_SIZE_BITS} -n 1)"
    gpufi_l1t_cache_bitflip_rand_n="${gpufi_l1t_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 instruction cache fault injections
    gpufi_l1i_shader_rand_n="$(shuf -e ${_SHADERS_USED} -n 1)"
    gpufi_l1i_shader_rand_n="${gpufi_l1i_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 instruction cache bit flips
    gpufi_l1i_cache_bitflip_rand_n="$(shuf -i 1-${L1I_SIZE_BITS} -n 1)"
    gpufi_l1i_cache_bitflip_rand_n="${gpufi_l1i_cache_bitflip_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L2 cache bit flips
    gpufi_l2_cache_bitflip_rand_n="$(shuf -i 1-${L2_SIZE_BITS} -n 1)"
    gpufi_l2_cache_bitflip_rand_n="${gpufi_l2_cache_bitflip_rand_n//$'\n'/:}"
    # ---------------------------------------------- END PER INJECTION CAMPAIGN PARAMETERS (_GPUFI_PROFILE=0) ------------------------------------------------

    sed -i -e "s/^-gpufi_components_to_flip.*$/-gpufi_components_to_flip ${COMPONENTS_TO_FLIP}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_profile.*$/-gpufi_profile ${_GPUFI_PROFILE}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_last_cycle.*$/-gpufi_last_cycle ${_TOTAL_CYCLES}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_thread_rand.*$/-gpufi_thread_rand ${gpufi_thread_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_warp_rand.*$/-gpufi_warp_rand ${gpufi_warp_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_total_cycle_rand.*$/-gpufi_total_cycle_rand ${gpufi_total_cycle_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_register_rand_n.*$/-gpufi_register_rand_n ${gpufi_register_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_reg_bitflip_rand_n.*$/-gpufi_reg_bitflip_rand_n ${gpufi_reg_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_per_warp.*$/-gpufi_per_warp ${PER_WARP}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_kernel_n.*$/-gpufi_kernel_n ${KERNEL_INDICES}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_local_mem_bitflip_rand_n.*$/-gpufi_local_mem_bitflip_rand_n ${gpufi_local_mem_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_block_rand.*$/-gpufi_block_rand ${gpufi_block_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_block_n.*$/-gpufi_block_n ${BLOCKS_NUM}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_shared_mem_bitflip_rand_n.*$/-gpufi_shared_mem_bitflip_rand_n ${gpufi_shared_mem_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1d_shader_rand_n.*$/-gpufi_l1d_shader_rand_n ${gpufi_l1d_shader_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1d_cache_bitflip_rand_n.*$/-gpufi_l1d_cache_bitflip_rand_n ${gpufi_l1d_cache_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1c_shader_rand_n.*$/-gpufi_l1c_shader_rand_n ${gpufi_l1c_shader_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1c_cache_bitflip_rand_n.*$/-gpufi_l1c_cache_bitflip_rand_n ${gpufi_l1c_cache_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1t_shader_rand_n.*$/-gpufi_l1t_shader_rand_n ${gpufi_l1t_shader_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1t_cache_bitflip_rand_n.*$/-gpufi_l1t_cache_bitflip_rand_n ${gpufi_l1t_cache_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1i_shader_rand_n.*$/-gpufi_l1i_shader_rand_n ${gpufi_l1i_shader_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l1i_cache_bitflip_rand_n.*$/-gpufi_l1i_cache_bitflip_rand_n ${gpufi_l1i_cache_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_l2_cache_bitflip_rand_n.*$/-gpufi_l2_cache_bitflip_rand_n ${gpufi_l2_cache_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s#^-gpufi_cycles_file.*\$#-gpufi_cycles_file ${_CYCLES_FILE}#" ${GPGPU_SIM_CONFIG_PATH}
}

_update_csv_file() {
    csv_file_path="$(_get_gpufi_analysis_path)/results.csv"
    run_id=$1
    # Turn 0 to 1 and the opposite
    success_msg_grep=$2
    success_msg_grep=$((success_msg_grep ^ 1))
    cycles_grep=$3
    cycles_grep=$((cycles_grep ^ 1))
    failed_msg_grep=$4
    failed_msg_grep=$((failed_msg_grep ^ 1))

    if [ ! -f "$csv_file_path" ]; then
        echo "run_id,success,same_cycles,failed" >"$csv_file_path"
    fi

    echo "${run_id},${success_msg_grep},${cycles_grep},${failed_msg_grep}" >>"$csv_file_path"

}

# Parses resulting logs and determines successful execution.
gather_results() {
    loop_num=$1 # Used to locate the logs of the run
    tmp_dir=${TMP_DIR}${loop_num}
    for batch_num in $(seq 1 $_NUM_AVAILABLE_CORES); do
        log_file="$tmp_dir/tmp.out${batch_num}"
        config_file="$tmp_dir/gpgpusim.config${batch_num}"
        echo "Examining file $log_file"
        grep -iq "${_SUCCESS_MSG}" "$log_file" && success_msg_grep=0 || success_msg_grep=1
        grep -i "${_CYCLES_MSG}" "$log_file" | tail -1 | grep -q "${_TOTAL_CYCLES}" && cycles_grep=0 || cycles_grep=1
        grep -iq "${_FAILED_MSG}" "$log_file" && failed_msg_grep=0 || failed_msg_grep=1

        # Result consists of three numbers:
        # - Was the _SUCCESS_MSG found in the resulting log?
        # - Were the total cycles same as the reference execution?
        # - Was the _FAILED_MSG found in the resulting log?
        result=${success_msg_grep}${cycles_grep}${failed_msg_grep}
        run_id=$(_calculate_md5_hash "$config_file" "$log_file" "$CUDA_EXECUTABLE_PATH" "$CUDA_EXECUTABLE_ARGS")
        if [ -n "$run_id" ]; then
            _update_csv_file $run_id $success_msg_grep $cycles_grep $failed_msg_grep
        fi
        case $result in
        "001")
            # Success msg found, same total cycles, no failure
            NUM_RUNS=$((NUM_RUNS - 1))
            _errors_masked=$((_errors_masked + 1))
            ;;
        "011")
            # Success msg found, different total cycles, no failure
            NUM_RUNS=$((NUM_RUNS - 1))
            _errors_masked=$((_errors_masked + 1))
            _errors_performance=$((_errors_performance + 1))
            ;;
        "100" | "110")
            # No success msg, same or different cycles, failure message
            NUM_RUNS=$((NUM_RUNS - 1))
            _errors_sdc=$((_errors_sdc + 1))
            ;;
        *)
            # Any other combination is considered a crash
            if grep -iq "${_FAULT_INJECTION_OCCURRED}" "$log_file"; then
                # Fault injection was performed, but then program crashed
                NUM_RUNS=$((NUM_RUNS - 1))
                _errors_due=$((_errors_due + 1))
                echo "Fault injection-related crash detected in loop $loop_num" # DEBUG
            else
                echo "Unclassified error in loop $loop_num: result=$result" # DEBUG
            fi

            ;;
        esac

    done
    echo "Finished gather_results"
}

# Execute a batch of batch_jobs in parallel and wait for them to finish.
# Once complete, parse their logs and increment the counters appropriately.
# For each batch, a new log directory is made with the loop number in it (e.g. logs3).
# In it, you should find one log file and one config file per batch job (1 -_NUM_AVAILABLE_CORES).
batch_execution() {
    batch_jobs=$1
    loop_num=$2
    tmp_dir=${TMP_DIR}${loop_num}
    echo "$(_get_timestamp): Loop $loop_num, batch jobs $batch_jobs"
    mkdir -p $tmp_dir >/dev/null 2>&1
    for i in $(seq 1 $batch_jobs); do
        echo "Starting loop $loop_num task $i/$batch_jobs"
        initialize_config
        # unique id for each run (e.g. r1b2: 1st run, 2nd execution on batch)
        sed -i -e "s/^-gpufi_run_id.*$/-gpufi_run_id r${loop_num}b${i}/" ${GPGPU_SIM_CONFIG_PATH}
        cp ${GPGPU_SIM_CONFIG_PATH} $tmp_dir/${GPGPU_SIM_CONFIG_PATH}${i} # save state
        # timeout $((_TIMEOUT_VALUE)) $CUDA_EXECUTABLE_PATH $CUDA_EXECUTABLE_ARGS >$tmp_dir/${TMP_FILE}${i} 2>&1 &
    done
    echo "Waiting for loop #$loop_num jobs to complete (total: $batch_jobs)"
    wait
    echo "Done"
    echo "Gathering results"
    gather_results $loop_num
    echo "Done"
    if [ $DELETE_LOGS -eq 1 ]; then
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt >/dev/null 2>&1
        rm -r $tmp_dir/${loop_num} >/dev/null 2>&1 # comment out to debug output
    fi
    if [ $_GPUFI_PROFILE -ne 1 ]; then
        # clean intermediate logs anyway if _GPUFI_PROFILE != 1
        rm -f _ptx* _cuobjdump_* _app_cuda* '*.ptx' f_tempfile_ptx gpgpu_inst_stats.txt >/dev/null 2>&1
    fi
}

# Main script function.
run_campaign() {
    if [[ "$_GPUFI_PROFILE" -eq 1 ]] || [[ "$_GPUFI_PROFILE" -eq 2 ]] || [[ "$_GPUFI_PROFILE" -eq 3 ]]; then
        NUM_RUNS=1
    fi
    # max_retries to avoid flooding the system storage with logs infinitely if the user
    # has wrong configuration and only Unclassified errors are returned.
    max_retries=$((3))
    current_loop_num=$((1))
    mkdir -p ${CACHE_LOGS_DIR} >/dev/null 2>&1
    while [ $NUM_RUNS -gt 0 ] && [ $max_retries -gt 0 ]; do
        max_retries=$((max_retries - 1))
        loop_start=$((current_loop_num))
        unset LAST_BATCH
        if [ "$_NUM_AVAILABLE_CORES" -gt "$NUM_RUNS" ]; then
            _NUM_AVAILABLE_CORES=${NUM_RUNS}
            loop_end=$loop_start
        else
            BATCH_RUNS=$((NUM_RUNS / _NUM_AVAILABLE_CORES))
            if ((NUM_RUNS % _NUM_AVAILABLE_CORES)); then
                LAST_BATCH=$((NUM_RUNS - BATCH_RUNS * _NUM_AVAILABLE_CORES))
            fi
            loop_end=$((loop_start + BATCH_RUNS - 1))
        fi

        for i in $(seq $loop_start $loop_end); do
            echo "Runs left: ${NUM_RUNS} (Loop $i/$loop_end)" # DEBUG
            batch_execution $_NUM_AVAILABLE_CORES $i
            current_loop_num=$((current_loop_num + 1))
        done

        if [[ -n ${LAST_BATCH+x} ]]; then
            batch_execution $LAST_BATCH $current_loop_num
            current_loop_num=$((current_loop_num + 1))
        fi
    done
    if [[ $max_retries -eq 0 ]]; then
        echo "Probably \"${CUDA_EXECUTABLE_PATH}\" was not able to run! Please make sure the execution with GPGPU-Sim works!"
    else
        echo "Masked: ${_errors_masked} (performance = ${_errors_performance})"
        echo "SDCs: ${_errors_sdc}"
        echo "DUEs: ${_errors_due}"
    fi
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -r ${CACHE_LOGS_DIR} >/dev/null 2>&1 # comment out to debug cache logs
    fi
}

read_params_from_gpgpusim_config() {
    source gpufi_calculate_cache_sizes.sh $GPGPU_SIM_CONFIG_PATH >/dev/null 2>&1
}

_get_gpufi_analysis_path() {
    echo "$(dirname $CUDA_EXECUTABLE_PATH)/.gpufi/$GPU_ID/$(_sanitize $CUDA_EXECUTABLE_ARGS)"
}

preliminary_checks() {
    if [ -z "$CUDA_EXECUTABLE_PATH" ]; then
        echo "Please provide a valid CUDA executable to run"
        exit 1
    fi

    if [ ! -f "$CUDA_EXECUTABLE_PATH" ]; then
        echo "File $CUDA_EXECUTABLE_PATH does not exist, please provide a valid executable"
        exit 1
    fi

    if [ -z "$GPGPU_SIM_CONFIG_PATH" ]; then
        echo "Please provide a valid gpgpusim.config"
        exit 1
    fi

    if [ ! -f "$GPGPU_SIM_CONFIG_PATH" ]; then
        echo "File $GPGPU_SIM_CONFIG_PATH does not exist, please provide a valid gpgpusim.config"
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
    # gpuFI TODO
    # if [ -z "$KERNEL_NAME" ]; then
    #     echo "Please provide a KERNEL_NAME to inject"
    #     exit 1
    # fi

    if [ ! -d "$(_get_gpufi_analysis_path)" ] && [ ! -f "$(_get_gpufi_analysis_path)/.analysis_complete" ]; then
        echo "Analysis not yet run for $CUDA_EXECUTABLE_PATH"
        exit 1
    fi

    if [ -z "$_GPUFI_PROFILE" ]; then
        echo "_GPUFI_PROFILE has not been selected. Setting to 0."
        _GPUFI_PROFILE=0
    fi

}

read_executable_analysis_files() {
    base_analysis_path=$(_get_gpufi_analysis_path)
    source "$base_analysis_path/executable_analysis.sh"
    if [ $KERNEL_INDICES -eq 0 ]; then
        source "$base_analysis_path/merged_kernel_analysis.sh"
        _CYCLES_FILE="$base_analysis_path/merged_cycles.txt"
        # gpuFI TODO: currently the merged files don't have SMEM, LMEM, registers...
    else
        source "$base_analysis_path/$KERNEL_NAME/kernel_analysis.sh"
        _CYCLES_FILE="$base_analysis_path/$KERNEL_NAME/cycles.txt"
    fi
}

### Script execution sequence ###
# Parse command line arguments -- use <key>=<value> to override any variable declared above.
for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"
    eval "$KEY=\"$VALUE\""
done

preliminary_checks
read_params_from_gpgpusim_config
read_executable_analysis_files
run_campaign

exit 0
