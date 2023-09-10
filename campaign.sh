#!/bin/bash

# ---------------------------------------------- START ONE-TIME PARAMETERS ----------------------------------------------
# needed by gpgpu-sim for real register usage on PTXPlus mode
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.2

CONFIG_FILE=./gpgpusim.config
TMP_DIR=./logs
CACHE_LOGS_DIR=./cache_logs
TMP_FILE=tmp.out
RUNS=7
BATCH=$(( $(grep -c ^processor /proc/cpuinfo) - 1 )) # -1 core for computer not to hang
DELETE_LOGS=0 # if 1 then all logs will be deleted at the end of the script
# ---------------------------------------------- END ONE-TIME PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER GPGPU CARD PARAMETERS ----------------------------------------------
source calculate_cache_sizes.sh
# ---------------------------------------------- END PER GPGPU CARD PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER KERNEL/APPLICATION PARAMETERS (+gpufi_profile=1) ----------------------------------------------
# Complete command for CUDA executable 
CUDA_UUT="./srad 2 0.5 128 128"

# total cycles for all kernels
CYCLES=49799

# Get the exact cycles, max registers and SIMT cores used for each kernel with gpufi_profile=1 
# fix cycles.txt with kernel execution cycles
# (e.g. seq 1 10 >> cycles.txt, or multiple seq commands if a kernel has multiple executions)
# use the following command from profiling execution for easier creation of cycles.txt file
# e.g. grep "_Z12lud_diagonalPfii" cycles.in | awk  '{ system("seq " $12 " " $18 ">> cycles.txt")}'
CYCLES_FILE=./cycles.txt
MAX_REGISTERS_USED=24
SHADER_USED="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 66 67 68 69 70 71 72 73 74 75 76 77 78 79"
SUCCESS_MSG='Test PASSED'
FAILED_MSG='Test FAILED'
TIMEOUT_VAL=400s

# Register size
DATATYPE_SIZE=32

# lmem and smem values are taken from gpgpu-sim ptx output per kernel
# e.g. GPGPU-Sim PTX: Kernel '_Z9vectorAddPKdS0_Pdi' : regs=8, lmem=0, smem=0, cmem=380
# if 0 put a random value > 0
LMEM_SIZE_BITS=10
SMEM_SIZE_BITS=1024
# ---------------------------------------------- END PER KERNEL/APPLICATION PARAMETERS (+gpufi_profile=1) ------------------------------------------------

FAULT_INJECTION_OCCURRED="Fault injection"
CYCLES_MSG="gpu_tot_sim_cycle ="

masked=0
performance=0
SDC=0
crashes=0

# ---------------------------------------------- START PER INJECTION CAMPAIGN PARAMETERS (gpufi_profile=0) ----------------------------------------------
# gpuFI profile to run. Possible values:
# 0: perform injection campaign
# 1: get cycles of each kernel
# 2: get mean value of active threads, during all cycles in CYCLES_FILE, per SM,
# 3: single fault-free execution
gpufi_profile=0

# Which components to apply a bif flip to. Multiple ones can be
# specified with a colon, e.g. gpufi_components_to_flip=0:1 for both RF and local_mem). Possible values:
# 0: RF
# 1: local_mem
# 2: shared_mem 
# 3: L1D_cache
# 4: L1C_cache
# 5: L1T_cache
# 6: L2_cache 
# 7: L1I 
gpufi_components_to_flip=0

# 1: per warp bit flip, 0: per thread bit flip
gpufi_per_warp=0

# in which kernels to inject the fault. e.g. 0: for all running kernels, 1: for kernel 1, 1:2 for kernel 1 & 2 
gpufi_kernel_n=0

# in how many blocks (smems) to inject the bit flip
blocks=1


# Function which initializes variables based on user config and edits the gpgpusim.config file.
initialize_config() {
    # random number for choosing a random thread after gpufi_thread_rand % #threads operation in gpgpu-sim
    gpufi_thread_rand=$(shuf -i 0-6000 -n 1)
    # random number for choosing a random warp after gpufi_warp_rand % #warp operation in gpgpu-sim
    gpufi_warp_rand=$(shuf -i 0-6000 -n 1)
    gpufi_total_cycle_rand=-1
    if [[ "$gpufi_profile" -eq 0 ]]; then
    # random cycle for fault injection
    gpufi_total_cycle_rand="$(shuf ${CYCLES_FILE} -n 1)"
    fi
    # in which registers to inject the bit flip
    gpufi_register_rand_n="$(shuf -i 1-${MAX_REGISTERS_USED} -n 1)"; gpufi_register_rand_n="${gpufi_register_rand_n//$'\n'/:}"
    # example: if -i 1-32 -n 2 then the two commands below will create a value with 2 random numbers, between [1,32] like 3:21. Meaning it will flip 3 and 21 bits.
    gpufi_reg_bitflip_rand_n="$(shuf -i 1-${DATATYPE_SIZE} -n 1)"; gpufi_reg_bitflip_rand_n="${gpufi_reg_bitflip_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for local memory bit flips
    gpufi_local_mem_bitflip_rand_n="$(shuf -i 1-${LMEM_SIZE_BITS} -n 3)"; gpufi_local_mem_bitflip_rand_n="${gpufi_local_mem_bitflip_rand_n//$'\n'/:}"
    # random number for choosing a random block after gpufi_block_rand % #smems operation in gpgpu-sim
    gpufi_block_rand=$(shuf -i 0-6000 -n 1)
    # same format like gpufi_reg_bitflip_rand_n but for shared memory bit flips
    gpufi_shared_mem_bitflip_rand_n="$(shuf -i 1-${SMEM_SIZE_BITS} -n 1)"; gpufi_shared_mem_bitflip_rand_n="${gpufi_shared_mem_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 data cache fault injections 
    gpufi_l1d_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"; gpufi_l1d_shader_rand_n="${gpufi_l1d_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 data cache bit flips
    gpufi_l1d_cache_bitflip_rand_n="$(shuf -i 1-${L1D_SIZE_BITS} -n 1)"; gpufi_l1d_cache_bitflip_rand_n="${gpufi_l1d_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 constant cache fault injections 
    gpufi_l1c_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"; gpufi_l1c_shader_rand_n="${gpufi_l1c_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 constant cache bit flips
    gpufi_l1c_cache_bitflip_rand_n="$(shuf -i 1-${L1C_SIZE_BITS} -n 1)"; gpufi_l1c_cache_bitflip_rand_n="${gpufi_l1c_cache_bitflip_rand_n//$'\n'/:}"
    # randomly select one or more shaders for L1 texture cache fault injections 
    gpufi_l1t_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"; gpufi_l1t_shader_rand_n="${gpufi_l1t_shader_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L1 texture cache bit flips
    gpufi_l1t_cache_bitflip_rand_n="$(shuf -i 1-${L1T_SIZE_BITS} -n 1)"; gpufi_l1t_cache_bitflip_rand_n="${gpufi_l1t_cache_bitflip_rand_n//$'\n'/:}"
    # same format like gpufi_reg_bitflip_rand_n but for L2 cache bit flips
    gpufi_l2_cache_bitflip_rand_n="$(shuf -i 1-${L2_SIZE_BITS} -n 1)"; gpufi_l2_cache_bitflip_rand_n="${gpufi_l2_cache_bitflip_rand_n//$'\n'/:}"
# ---------------------------------------------- END PER INJECTION CAMPAIGN PARAMETERS (gpufi_profile=0) ------------------------------------------------

    sed -i -e "s/^-gpufi_components_to_flip.*$/-gpufi_components_to_flip ${gpufi_components_to_flip}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_profile.*$/-gpufi_profile ${gpufi_profile}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_last_cycle.*$/-gpufi_last_cycle ${CYCLES}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_thread_rand.*$/-gpufi_thread_rand ${gpufi_thread_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_warp_rand.*$/-gpufi_warp_rand ${gpufi_warp_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_total_cycle_rand.*$/-gpufi_total_cycle_rand ${gpufi_total_cycle_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_register_rand_n.*$/-gpufi_register_rand_n ${gpufi_register_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_reg_bitflip_rand_n.*$/-gpufi_reg_bitflip_rand_n ${gpufi_reg_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_per_warp.*$/-gpufi_per_warp ${gpufi_per_warp}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_kernel_n.*$/-gpufi_kernel_n ${gpufi_kernel_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_local_mem_bitflip_rand_n.*$/-gpufi_local_mem_bitflip_rand_n ${gpufi_local_mem_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_block_rand.*$/-gpufi_block_rand ${gpufi_block_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_block_n.*$/-gpufi_block_n ${blocks}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_shared_mem_bitflip_rand_n.*$/-gpufi_shared_mem_bitflip_rand_n ${gpufi_shared_mem_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-shader_rand_n.*$/-shader_rand_n ${shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l1d_shader_rand_n.*$/-gpufi_l1d_shader_rand_n ${gpufi_l1d_shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l1d_cache_bitflip_rand_n.*$/-gpufi_l1d_cache_bitflip_rand_n ${gpufi_l1d_cache_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l1c_shader_rand_n.*$/-gpufi_l1c_shader_rand_n ${gpufi_l1c_shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l1c_cache_bitflip_rand_n.*$/-gpufi_l1c_cache_bitflip_rand_n ${gpufi_l1c_cache_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l1t_shader_rand_n.*$/-gpufi_l1t_shader_rand_n ${gpufi_l1t_shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l1t_cache_bitflip_rand_n.*$/-gpufi_l1t_cache_bitflip_rand_n ${gpufi_l1t_cache_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-gpufi_l2_cache_bitflip_rand_n.*$/-gpufi_l2_cache_bitflip_rand_n ${gpufi_l2_cache_bitflip_rand_n}/" ${CONFIG_FILE}
}

# Parses resulting logs and determines successful execution.
gather_results() {
    for file in ${TMP_DIR}${1}/${TMP_FILE}*; do
        if [[ "$gpufi_profile" -eq 1 ]]; then
            # Find start and end cycles for each kernel  
            grep -E "Kernel = [[:digit:]]+.+" $file | sort -t' ' -k 3 -g > ${TMP_DIR}${1}/cycles.in
            # TODO: parse Kernel = %s, max active regs = %u
            # TODO: parse Kernel = %s used shaders
        fi
        grep -iq "${SUCCESS_MSG}" $file; success_msg_grep=$(echo $?)
	grep -i "${CYCLES_MSG}" $file | tail -1 | grep -q "${CYCLES}"; cycles_grep=$(echo $?)
        grep -iq "${FAILED_MSG}" $file; failed_msg_grep=$(echo $?)
        result=${success_msg_grep}${cycles_grep}${failed_msg_grep}
        case $result in
        "001")
            let RUNS--
            let masked++ ;;
        "011")
            let RUNS--
            let masked++ 
            let performance++ ;;
        "100" | "110")
            let RUNS--
            let SDC++ ;;
        *)
            grep -iq "${FAULT_INJECTION_OCCURRED}" $file
            if [ $? -eq 0 ]; then
                let RUNS--
                let crashes++
                echo "Crash appeared in loop ${1}" # DEBUG
            else
                echo "Unclassified in loop ${1} ${result}" # DEBUG
            fi ;;
        esac
    done
}

parallel_execution() {
    batch=$1
    mkdir -p ${TMP_DIR}${2} > /dev/null 2>&1
    for i in $( seq 1 $batch ); do
        initialize_config
        # unique id for each run (e.g. r1b2: 1st run, 2nd execution on batch)
        sed -i -e "s/^-gpufi_run_id.*$/-gpufi_run_id r${2}b${i}/" ${CONFIG_FILE}
        cp ${CONFIG_FILE} ${TMP_DIR}${2}/${CONFIG_FILE}${i} # save state
        timeout ${TIMEOUT_VAL} $CUDA_UUT > ${TMP_DIR}${2}/${TMP_FILE}${i} 2>&1 &
    done
    wait
    gather_results $2
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt > /dev/null 2>&1
        rm -r ${TMP_DIR}${2} > /dev/null 2>&1 # comment out to debug output
    fi
    if [[ "$gpufi_profile" -ne 1 ]]; then
        # clean intermediate logs anyway if gpufi_profile != 1
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt > /dev/null 2>&1
    fi
}

# Main script entrypoint. Gets passed all the parameters that are passed to the script itself.
main() {
    if [[ "$gpufi_profile" -eq 1 ]] || [[ "$gpufi_profile" -eq 2 ]] || [[ "$gpufi_profile" -eq 3 ]]; then
        RUNS=1
    fi
    # MAX_RETRIES to avoid flooding the system storage with logs infinitely if the user
    # has wrong configuration and only Unclassified errors are returned
    MAX_RETRIES=3
    LOOP=1
    mkdir -p ${CACHE_LOGS_DIR} > /dev/null 2>&1
    while [[ $RUNS -gt 0 ]] && [[ $MAX_RETRIES -gt 0 ]]
    do
        echo "runs left ${RUNS}" # DEBUG
        let MAX_RETRIES--
        LOOP_START=${LOOP}
        unset LAST_BATCH
        if [ "$BATCH" -gt "$RUNS" ]; then
            BATCH=${RUNS}
            LOOP_END=$(($LOOP_START))
        else
            BATCH_RUNS=$(($RUNS/$BATCH))
            if (( $RUNS % $BATCH )); then
                LAST_BATCH=$(($RUNS-$BATCH_RUNS*$BATCH))
            fi
            LOOP_END=$(($LOOP_START+$BATCH_RUNS-1))
        fi

        for i in $( seq $LOOP_START $LOOP_END ); do
            parallel_execution $BATCH $i
            let LOOP++
        done

        if [[ ! -z ${LAST_BATCH+x} ]]; then
            parallel_execution $LAST_BATCH $LOOP
            let LOOP++
        fi
    done

    if [[ $MAX_RETRIES -eq 0 ]]; then
        echo "Probably \"${CUDA_UUT}\" was not able to run! Please make sure the execution with GPGPU-Sim works!"
    else
        echo "Masked: ${masked} (performance = ${performance})"
        echo "SDCs: ${SDC}"
        echo "DUEs: ${crashes}"
    fi
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -r ${CACHE_LOGS_DIR} > /dev/null 2>&1 # comment out to debug cache logs
    fi
}

main "$@"
exit 0

