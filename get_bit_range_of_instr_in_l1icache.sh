#!/bin/bash

# gpUFI helper script for locating the approximate bits of a specific instruction in cache.
# Intended for debugging purposes.
# Takes two arguments:
# 1. PC (in HEX, e.g. "5af") of the instruction that you want to locate in the L1I cache.
# 2. PC (in HEX) of the last instruction of the executable.
# Assumes n-way associative cache.

# Program counter of instruction to inject.
PC=$(echo "ibase=16;$(echo $1 | tr '[:lower:]' '[:upper:]')" | bc)
# PC of last instruction in program's SASS dump.
LAST_PC=$(echo "ibase=16;$(echo $2 | tr '[:lower:]' '[:upper:]')" | bc)
# gpuFI TODO: Verify that neigher PC or LAST_PC are empty.

# Instruction address in global mem, taking PROGRAM_MEM_START into account.
# PC's are offset by the PC of the program's last instruction.
PROGRAM_MEM_START=$(echo "ibase=16;F0000000" | bc)

# This may not always work; Sometimes there is even more offset (by several instructions,
# e.g. 16 bytes). This may mess up the calculations. The best way to get the actual PC of the
# instruction is to run the simulator with a debugger and set a breakpoint on
# gpgpu_context::pc_to_instruction(). Then, inspect s_g_pc_to_insn. Then, rerun the script
# by adjusting the PC with the offset you found.
INSTR_SIMULATOR_PC=$((PC + LAST_PC))
echo "Instruction's PC in the simulator may be $INSTR_SIMULATOR_PC"

INSTR_ADDR=$((INSTR_SIMULATOR_PC + PROGRAM_MEM_START))
echo "Instruction with PC 0x$1 will be stored in address $INSTR_ADDR"

# Do calculations of cache sizes
source ./calculate_cache_sizes.sh >/dev/null

# Offset of instruction in line
li1_line_byte_offset=$((INSTR_ADDR & (L1I_BYTES_PER_LINE - 1)))
echo "Instr byte offset: $li1_line_byte_offset"
# Index of the set where the instruction will be cached
l1i_set_idx=$(((INSTR_ADDR >> L1I_BITS_FOR_BYTE_OFFSET) & (L1I_NUM_SETS - 1)))
# echo "Address $INSTR_ADDR will be cached in set $l1i_set_idx"

# GPGPU Sim stores a continuous array of "lines": find the index of the C++ array
l1i_line_idx=$((l1i_set_idx * L1I_ASSOC))
echo "Instruction with pc 0x$PC can be found in L1I cache between lines $l1i_line_idx and $((l1i_line_idx + (L1I_ASSOC - 1)))"
echo "Bit ranges that the instructions may be found, per each cache line:"

bit_line_start=$((l1i_line_idx * (L1I_TAG_BITS + L1I_BYTES_PER_LINE * 8) + L1I_TAG_BITS))
for i in $(seq $l1i_line_idx $((l1i_line_idx + L1I_ASSOC - 1))); do
    bit_line_end=$((bit_line_start + L1I_BYTES_PER_LINE * 8))
    # echo "Line bit range: $bit_line_start - $bit_line_end"
    bit_instr_start=$((1 + bit_line_start + li1_line_byte_offset * 8))
    bit_instr_end=$((bit_instr_start + 63)) # Not always the case, some instructions are half-size
    echo $i: $bit_instr_start - $bit_instr_end
    bit_line_start=$((bit_line_end + L1I_TAG_BITS))
done
