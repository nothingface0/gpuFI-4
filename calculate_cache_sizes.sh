#!/bin/bash

# This file calculates the total size (in bits) of L1D, L1T, L1C and L2 cache.
ADDRESS_WIDTH=64

# ---------------- L1D calculations ------------------
regex_l1d="^-gpgpu_cache:dl1[[:space:]]*[[:alpha:]]?:?([0-9]+):([0-9]+):([0-9]+),"

# Find L1D config in gpgpusim.config w/ gawk
parsed_regex=$(cat gpgpusim.config | gawk -v pat=$regex_l1d 'match($0, pat, a) {print a[1], a[2], a[3]}')
parsed_arr=(${parsed_regex// / })  # Split with spaces

l1d_sets=${parsed_arr[0]}
# Use bc to calculate log_2, then printf to remove decimals
l1d_bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($l1d_sets)/l(2)") )

l1d_bytes_per_line=${parsed_arr[1]}
l1d_bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($l1d_bytes_per_line)/l(2)") )

l1d_associativity=${parsed_arr[2]}

echo "L1D: sets="$l1d_sets", bytes per line="$l1d_bytes_per_line", associativity="$l1d_associativity

l1d_tag_bits=$(($ADDRESS_WIDTH-$l1d_bits_for_byte_offset-$l1d_bits_for_sets_indexing))
echo "Bits for tag (L1D)="$l1d_tag_bits
echo "Bits for tag+index (L1D)="$(($l1d_tag_bits+$l1d_bits_for_sets_indexing))

l1d_total_bits=$(( ($l1d_bytes_per_line*8+$l1d_tag_bits)*$l1d_associativity*$l1d_sets ))
echo "L1D size per SIMT core (bits)="$l1d_total_bits
echo 

# ---------------- L1T calculations ------------------
regex_l1t="^-gpgpu_tex_cache:l1[[:space:]]*[[:alpha:]]?:?([0-9]+):([0-9]+):([0-9]+),"

# Find L1T config in gpgpusim.config w/ gawk
parsed_regex=$(cat gpgpusim.config | gawk -v pat=$regex_l1t 'match($0, pat, a) {print a[1], a[2], a[3]}')
parsed_arr=(${parsed_regex// / })  # Split with spaces

l1t_sets=${parsed_arr[0]}
# Use bc to calculate log_2, then printf to remove decimals
l1t_bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($l1t_sets)/l(2)") )

l1t_bytes_per_line=${parsed_arr[1]}
l1t_bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($l1t_bytes_per_line)/l(2)") )

l1t_associativity=${parsed_arr[2]}

echo "L1T: sets="$l1t_sets", bytes per line="$l1t_bytes_per_line", associativity="$l1t_associativity

l1t_tag_bits=$(($ADDRESS_WIDTH-$l1t_bits_for_byte_offset-$l1t_bits_for_sets_indexing))
echo "Bits for tag (L1T)="$l1t_tag_bits
echo "Bits for tag+index (L1T)="$(($l1t_tag_bits+$l1t_bits_for_sets_indexing))

l1t_total_bits=$(( ($l1t_bytes_per_line*8+$l1t_tag_bits)*$l1t_associativity*$l1d_sets ))
echo "L1T size per SIMT core (bits)="$l1t_total_bits
echo 

# ---------------- L1C calculations ------------------
regex_l1c="^-gpgpu_const_cache:l1[[:space:]]*[[:alpha:]]?:?([0-9]+):([0-9]+):([0-9]+),"

# Find L1C config in gpgpusim.config w/ gawk
parsed_regex=$(cat gpgpusim.config | gawk -v pat=$regex_l1c 'match($0, pat, a) {print a[1], a[2], a[3]}')
parsed_arr=(${parsed_regex// / })  # Split with spaces

l1c_sets=${parsed_arr[0]}
# Use bc to calculate log_2, then printf to remove decimals
l1c_bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($l1c_sets)/l(2)") )

l1c_bytes_per_line=${parsed_arr[1]}
l1c_bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($l1c_bytes_per_line)/l(2)") )

l1c_associativity=${parsed_arr[2]}

echo "L1C: sets="$l1c_sets", bytes per line="$l1c_bytes_per_line", associativity="$l1c_associativity

l1c_tag_bits=$(($ADDRESS_WIDTH-$l1c_bits_for_byte_offset-$l1c_bits_for_sets_indexing))
echo "Bits for tag (L1C)="$l1c_tag_bits
echo "Bits for tag+index (L1C)="$(($l1c_tag_bits+$l1c_bits_for_sets_indexing))

l1c_total_bits=$(( ($l1c_bytes_per_line*8+$l1c_tag_bits)*$l1c_associativity*$l1c_sets ))
echo "L1C size per SIMT core (bits)="$l1c_total_bits
echo 

# ---------------- L2 calculations ------------------
regex_l2d="^-gpgpu_cache:dl2[[:space:]]*[[:alpha:]]?:?([0-9]+):([0-9]+):([0-9]+),"
regex_num_mem_controllers="^-gpgpu_n_mem[[:space:]]*([0-9]+)"
regex_num_sub_partitions="^-gpgpu_n_sub_partition_per_mchannel[[:space:]]*([0-9]+)"

# Find L2D config in gpgpusim.config w/ gawk
parsed_regex=$(cat gpgpusim.config | gawk -v pat=$regex_l2d 'match($0, pat, a) {print a[1], a[2], a[3]}')
parsed_arr=(${parsed_regex// / }) # Split with spaces
l2d_sets=${parsed_arr[0]}
l2d_bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($l2d_sets)/l(2)") )

l2d_bytes_per_line=${parsed_arr[1]}
l2d_bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($l2d_bytes_per_line)/l(2)") )

l2d_associativity=${parsed_arr[2]}

echo "l2d: sets="$l2d_sets", bytes per line="$l2d_bytes_per_line", associativity="$l2d_associativity

l2d_tag_bits=$(($ADDRESS_WIDTH-$l2d_bits_for_byte_offset-$l2d_bits_for_sets_indexing))
echo "Bits for tag (L2D)="$l2d_tag_bits
echo "Bits for tag+index (L2D)="$(($l2d_tag_bits+$l2d_bits_for_sets_indexing))

l2d_num_mem_controllers=$(cat gpgpusim.config | gawk -v pat=$regex_num_mem_controllers 'match($0, pat, a) {print a[1]}')
l2d_num_sub_partitions=$(cat gpgpusim.config | gawk -v pat=$regex_num_sub_partitions 'match($0, pat, a) {print a[1]}')

# There's one L2D cache per memory subpartition, each one having the 
# configuration given by the gpgpu_cache:dl2 option.
# Having calculated the tag bits, we can calculate the total L2 cache
# size in bits as follows:
l2d_total_bits=$(( ($l2d_bytes_per_line*8+$l2d_tag_bits)*$l2d_associativity*$l2d_sets*$l2d_num_mem_controllers*$l2d_num_sub_partitions ))
echo "L2D total size (bits)="$l2d_total_bits

# --------- Aliases for the rest of campaign.sh
L1D_SIZE_BITS=$l1d_total_bits
L1C_SIZE_BITS=$l1c_total_bits
L1T_SIZE_BITS=$l1t_total_bits
L2_SIZE_BITS=$l2d_total_bits