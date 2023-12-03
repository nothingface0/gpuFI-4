#!/bin/bash

# gpuFI script for running an injection campaign for a combination of:
# - A specific gpgpusim.config (i.e. a specific GPU),
# - A specific CUDA executable and its arguments, and
# - A kernel of the executable to be targeted.

# set -x
source gpufi_utils.sh
# Get files created from gpufi_analyze_executable.sh
EXECUTABLE_ANALYSIS_FILE=
KERNEL_ANALYSIS_FILE=
GPU_ID=
KERNEL_NAME=
# ---------------------------------------------- START ONE-TIME PARAMETERS ----------------------------------------------
# needed by gpgpu-sim for real register usage on PTXPlus mode
#export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.0

GPGPU_SIM_CONFIG_PATH=
TMP_DIR=./logs
CACHE_LOGS_DIR=./cache_logs
TMP_FILE=tmp.out
NUM_RUNS=7
NUM_AVAILABLE_CORES=$(($(nproc) - 1)) # -1 core for computer not to hang
DELETE_LOGS=0                         # if 1 then all logs will be deleted at the end of the script
# ---------------------------------------------- END ONE-TIME PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER GPGPU CARD PARAMETERS ----------------------------------------------

# ---------------------------------------------- END PER GPGPU CARD PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER KERNEL/APPLICATION PARAMETERS (+GPUFI_PROFILE=1) ----------------------------------------------
# Register size
DATATYPE_SIZE=32

# Complete command for CUDA executable
CUDA_EXECUTABLE_PATH=
CUDA_EXECUTABLE_ARGS=""

# total cycles for all kernels
TOTAL_CYCLES=
# Time to wait for executable to finish running.
TIMEOUT_VALUE=

# Per kernel config
CYCLES_FILE=
MAX_REGISTERS_USED=
SHADERS_USED=
SUCCESS_MSG='Test PASSED'
FAILED_MSG='Test FAILED'

# lmem and smem values are taken from gpgpu-sim ptx output per kernel
# e.g. GPGPU-Sim PTX: Kernel '_Z9vectorAddPKdS0_Pdi' : regs=8, lmem=0, smem=0, cmem=380
# if 0 put a random value > 0
LMEM_SIZE_BITS=
SMEM_SIZE_BITS=
# ---------------------------------------------- END PER KERNEL/APPLICATION PARAMETERS (+GPUFI_PROFILE=1) ------------------------------------------------

FAULT_INJECTION_OCCURRED="gpuFI: Fault injection"
CYCLES_MSG="gpu_tot_sim_cycle ="

masked=0
performance=0
SDC=0
num_crashes=0

# ---------------------------------------------- START PER INJECTION CAMPAIGN PARAMETERS (GPUFI_PROFILE=0) ----------------------------------------------
# gpuFI profile to run. Possible values:
# 0: perform injection campaign
# 1: get cycles of each kernel
# 2: get mean value of active threads, during all cycles in CYCLES_FILE, per SM,
# 3: single fault-free execution
GPUFI_PROFILE=

# Which components to apply a bif flip to. Multiple ones can be
# specified with a colon, e.g. gpufi_components_to_flip=0:1 for both RF and local_mem). Possible values:
# 0: RF
# 1: local_mem
# 2: shared_mem
# 3: L1D_cache
# 4: L1C_cache
# 5: L1T_cache
# 6: L2_cache
# 7: L1I_cache
gpufi_components_to_flip=7

# 0: per thread bit flip, 1: per warp bit flip
gpufi_per_warp=0

# in which kernels to inject the fault. e.g. 0: for all running kernels, 1: for kernel 1, 1:2 for kernel 1 & 2
gpufi_kernel_n=0

# in how many blocks (smems) to inject the bit flip
gpufi_block_n=1

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
    if [[ "$GPUFI_PROFILE" -eq 0 ]]; then
        # random cycle for fault injection
        gpufi_total_cycle_rand="$(shuf ${CYCLES_FILE} -n 1)"
    fi
    # in which registers to inject the bit flip
    gpufi_register_rand_n="$(shuf -i 1-${MAX_REGISTERS_USED} -n 1)"
    gpufi_register_rand_n="${gpufi_register_rand_n//$'\n'/:}"
    # example: if -i 1-32 -n 2 then the two commands below will create a value with 2 random numbers, between [1,32] like 3:21. Meaning it will flip 3 and 21 bits.
    gpufi_reg_bitflip_rand_n="$(shuf -i 1-${DATATYPE_SIZE} -n 1)"
    gpufi_reg_bitflip_rand_n="${gpufi_reg_bitflip_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for local memory bit flips
    gpufi_local_mem_bitflip_rand_n="$(shuf -i 1-${LMEM_SIZE_BITS} -n 3)"
    gpufi_local_mem_bitflip_rand_n="${gpufi_local_mem_bitflip_rand_n//$'\n'/:}"
    # random number for choosing a random block after gpufi_block_rand % #smems operation in gpgpu-sim
    gpufi_block_rand=$(shuf -i 0-6000 -n 1)
    # same format like gpufi_reg_bitflip_rand_n but for shared memory bit flips
    gpufi_shared_mem_bitflip_rand_n="$(shuf -i 1-${SMEM_SIZE_BITS} -n 1)"
    gpufi_shared_mem_bitflip_rand_n="${gpufi_shared_mem_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 data cache fault injections
    gpufi_l1d_shader_rand_n="$(shuf -e ${SHADERS_USED} -n 1)"
    gpufi_l1d_shader_rand_n="${gpufi_l1d_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 data cache bit flips
    gpufi_l1d_cache_bitflip_rand_n="$(shuf -i 1-${L1D_SIZE_BITS} -n 1)"
    gpufi_l1d_cache_bitflip_rand_n="${gpufi_l1d_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 constant cache fault injections
    gpufi_l1c_shader_rand_n="$(shuf -e ${SHADERS_USED} -n 1)"
    gpufi_l1c_shader_rand_n="${gpufi_l1c_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 constant cache bit flips
    gpufi_l1c_cache_bitflip_rand_n="$(shuf -i 1-${L1C_SIZE_BITS} -n 1)"
    gpufi_l1c_cache_bitflip_rand_n="${gpufi_l1c_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 texture cache fault injections
    gpufi_l1t_shader_rand_n="$(shuf -e ${SHADERS_USED} -n 1)"
    gpufi_l1t_shader_rand_n="${gpufi_l1t_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 texture cache bit flips
    gpufi_l1t_cache_bitflip_rand_n="$(shuf -i 1-${L1T_SIZE_BITS} -n 1)"
    gpufi_l1t_cache_bitflip_rand_n="${gpufi_l1t_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 instruction cache fault injections
    gpufi_l1i_shader_rand_n="$(shuf -e ${SHADERS_USED} -n 1)"
    gpufi_l1i_shader_rand_n="${gpufi_l1i_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 instruction cache bit flips
    gpufi_l1i_cache_bitflip_rand_n="$(shuf -i 1-${L1I_SIZE_BITS} -n 1)"
    gpufi_l1i_cache_bitflip_rand_n="${gpufi_l1i_cache_bitflip_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L2 cache bit flips
    gpufi_l2_cache_bitflip_rand_n="$(shuf -i 1-${L2_SIZE_BITS} -n 1)"
    gpufi_l2_cache_bitflip_rand_n="${gpufi_l2_cache_bitflip_rand_n//$'\n'/:}"
    # ---------------------------------------------- END PER INJECTION CAMPAIGN PARAMETERS (GPUFI_PROFILE=0) ------------------------------------------------

    sed -i -e "s/^-gpufi_components_to_flip.*$/-gpufi_components_to_flip ${gpufi_components_to_flip}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_profile.*$/-gpufi_profile ${GPUFI_PROFILE}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_last_cycle.*$/-gpufi_last_cycle ${TOTAL_CYCLES}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_thread_rand.*$/-gpufi_thread_rand ${gpufi_thread_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_warp_rand.*$/-gpufi_warp_rand ${gpufi_warp_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_total_cycle_rand.*$/-gpufi_total_cycle_rand ${gpufi_total_cycle_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_register_rand_n.*$/-gpufi_register_rand_n ${gpufi_register_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_reg_bitflip_rand_n.*$/-gpufi_reg_bitflip_rand_n ${gpufi_reg_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_per_warp.*$/-gpufi_per_warp ${gpufi_per_warp}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_kernel_n.*$/-gpufi_kernel_n ${gpufi_kernel_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_local_mem_bitflip_rand_n.*$/-gpufi_local_mem_bitflip_rand_n ${gpufi_local_mem_bitflip_rand_n}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_block_rand.*$/-gpufi_block_rand ${gpufi_block_rand}/" ${GPGPU_SIM_CONFIG_PATH}
    sed -i -e "s/^-gpufi_block_n.*$/-gpufi_block_n ${gpufi_block_n}/" ${GPGPU_SIM_CONFIG_PATH}
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
}

# Parses resulting logs and determines successful execution.
gather_results() {
    for file in ${TMP_DIR}${1}/${TMP_FILE}*; do
        # Done in gpufi_analyze_executable.sh
        # if [[ "$GPUFI_PROFILE" -eq 1 ]]; then
        #     # Find start and end cycles for each kernel
        #     grep -E "gpuFI: Kernel = [[:digit:]]+.+" $file | sort -t' ' -k 3 -g >${TMP_DIR}${1}/cycles.in
        #     # TODO: parse Kernel = %s, max active regs = %u
        #     # TODO: parse Kernel = %s used shaders
        # fi
        grep -iq "${SUCCESS_MSG}" $file
        success_msg_grep=$?
        grep -i "${CYCLES_MSG}" $file | tail -1 | grep -q "${TOTAL_CYCLES}"
        cycles_grep=$?
        grep -iq "${FAILED_MSG}" $file
        failed_msg_grep=$?
        # Result consists of three numbers:
        # - Was the SUCCESS_MSG found in the resulting log?
        # - Were the total cycles same as the reference execution?
        # - Was the FAILED_MSG found in the resulting log?
        result=${success_msg_grep}${cycles_grep}${failed_msg_grep}
        case $result in
        "001")
            # Success msg found, same total cycles, no failure
            ((NUM_RUNS--))
            ((masked++))
            ;;
        "011")
            # Success msg found, different total cycles, no failure
            ((NUM_RUNS--))
            ((masked++))
            ((performance++))
            ;;
        "100" | "110")
            # No success msg, same or different cycles, failure message
            ((NUM_RUNS--))
            ((SDC++))
            ;;
        *)
            # Any other combination is considered a crash
            if grep -iq "${FAULT_INJECTION_OCCURRED}" "$file"; then
                # Fault injection was performed, but then program crashed
                ((NUM_RUNS--))
                ((num_crashes++))
                echo "Fault injection-related crash detected in loop ${1}" # DEBUG
            else
                echo "Unclassified error in loop ${1}: result=${result}" # DEBUG
            fi
            ;;
        esac
    done
}

parallel_execution() {
    batch=$1
    mkdir -p ${TMP_DIR}${2} >/dev/null 2>&1
    for i in $(seq 1 $batch); do
        initialize_config
        # unique id for each run (e.g. r1b2: 1st run, 2nd execution on batch)
        sed -i -e "s/^-gpufi_run_id.*$/-gpufi_run_id r${2}b${i}/" ${GPGPU_SIM_CONFIG_PATH}
        cp ${GPGPU_SIM_CONFIG_PATH} ${TMP_DIR}${2}/${GPGPU_SIM_CONFIG_PATH}${i} # save state
        timeout ${TIMEOUT_VALUE} $CUDA_EXECUTABLE_PATH >${TMP_DIR}${2}/${TMP_FILE}${i} 2>&1 &
    done
    wait
    gather_results $2
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt >/dev/null 2>&1
        rm -r ${TMP_DIR}${2} >/dev/null 2>&1 # comment out to debug output
    fi
    if [[ "$GPUFI_PROFILE" -ne 1 ]]; then
        # clean intermediate logs anyway if GPUFI_PROFILE != 1
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt >/dev/null 2>&1
    fi
}

# Main script function.
main() {
    if [[ "$GPUFI_PROFILE" -eq 1 ]] || [[ "$GPUFI_PROFILE" -eq 2 ]] || [[ "$GPUFI_PROFILE" -eq 3 ]]; then
        NUM_RUNS=1
    fi
    # max_retries to avoid flooding the system storage with logs infinitely if the user
    # has wrong configuration and only Unclassified errors are returned
    max_retries=3
    current_loop_num=1
    mkdir -p ${CACHE_LOGS_DIR} >/dev/null 2>&1
    while [[ $NUM_RUNS -gt 0 ]] && [[ $max_retries -gt 0 ]]; do
        echo "runs left ${NUM_RUNS}" # DEBUG
        ((max_retries--))
        loop_start=${current_loop_num}
        unset LAST_BATCH
        if [ "$NUM_AVAILABLE_CORES" -gt "$NUM_RUNS" ]; then
            NUM_AVAILABLE_CORES=${NUM_RUNS}
            loop_end=$loop_start
        else
            BATCH_RUNS=$((NUM_RUNS / NUM_AVAILABLE_CORES))
            if ((NUM_RUNS % NUM_AVAILABLE_CORES)); then
                LAST_BATCH=$((NUM_RUNS - BATCH_RUNS * NUM_AVAILABLE_CORES))
            fi
            loop_end=$((loop_start + BATCH_RUNS - 1))
        fi

        for i in $(seq $loop_start $loop_end); do
            parallel_execution $NUM_AVAILABLE_CORES $i
            ((current_loop_num++))
        done

        if [[ -n ${LAST_BATCH+x} ]]; then
            parallel_execution $LAST_BATCH $current_loop_num
            ((current_loop_num++))
        fi
    done

    if [[ $max_retries -eq 0 ]]; then
        echo "Probably \"${CUDA_EXECUTABLE_PATH}\" was not able to run! Please make sure the execution with GPGPU-Sim works!"
    else
        echo "Masked: ${masked} (performance = ${performance})"
        echo "SDCs: ${SDC}"
        echo "DUEs: ${num_crashes}"
    fi
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -r ${CACHE_LOGS_DIR} >/dev/null 2>&1 # comment out to debug cache logs
    fi
}

read_gpgpusim_config() {
    source gpufi_calculate_cache_sizes.sh $GPGPU_SIM_CONFIG_PATH
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
    if [ -z "$KERNEL_NAME" ]; then
        echo "Please provide a KERNEL_NAME to inject"
        exit 1
    fi

    if [ ! -d "$(_get_gpufi_analysis_path)" ] && [ ! -f "$(_get_gpufi_analysis_path)/.analysis_complete" ]; then
        echo "Analysis not yet run for $CUDA_EXECUTABLE_PATH"
        exit 1
    fi

    if [ -z "$GPUFI_PROFILE" ]; then
        echo "GPUFI_PROFILE has not been selected. Setting to 0."
        GPUFI_PROFILE=0
    fi

}

read_executable_analysis_files() {
    base_analysis_path=$(_get_gpufi_analysis_path)
    source "$base_analysis_path/executable_analysis.sh"
    source "$base_analysis_path/$KERNEL_NAME/kernel_analysis.sh"
    CYCLES_FILE="$base_analysis_path/$KERNEL_NAME/cycles.txt"
}

### Script execution sequence ###
set -x
# Parse command line arguments -- use <key>=<value> to override the flags mentioned above.
for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"
    eval "$KEY=\"$VALUE\""
done

preliminary_checks
read_gpgpusim_config
read_executable_analysis_files
exit 0
main
exit 0
