#!/bin/bash
set -e
# This file calculates the total size (in bits) of L1D, L1T, L1C and L2 cache.
ADDRESS_WIDTH=64
WORD_SIZE=64
CONFIG_FILE=./gpgpusim.config
if [[ ! -z $1 ]] 
then
    if test -f $1; then
        CONFIG_FILE=$1
    fi;
fi;

l1_cache_bits_and_size_calculations () {
    cache_name=$1
    cache_config_id=$2
    regex="^$cache_config_id[[:space:]]*[[:alpha:]]?:?([0-9]+):([0-9]+):([0-9]+),"

    # Find config in gpgpusim.config w/ gawk
    parsed_regex=$(cat $CONFIG_FILE | gawk -v pat=$regex 'match($0, pat, a) {print a[1], a[2], a[3]}')
    parsed_arr=(${parsed_regex// / })  # Split with spaces

    sets=${parsed_arr[0]}
    # Use bc to calculate log_2, then printf to remove decimals
    bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($sets)/l(2)") )

    bytes_per_line=${parsed_arr[1]}
    bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($bytes_per_line)/l(2)") )
    bits_for_word_offset=$( printf "%.0f" $(bc -l <<< "l($bytes_per_line/($WORD_SIZE/8))/l(2)") )

    associativity=${parsed_arr[2]}

    echo "$cache_name: sets="$sets", bytes per line="$bytes_per_line", associativity="$associativity

    tag_bits=$(($ADDRESS_WIDTH-$bits_for_byte_offset-$bits_for_sets_indexing))
    echo "Bits for tag="$tag_bits
    echo "Bits for index="$bits_for_sets_indexing
    echo "Bits for byte offset="$bits_for_byte_offset
    echo "Bits for word offset="$bits_for_word_offset
    echo "Bits for tag+index="$(($tag_bits+$bits_for_sets_indexing))

    total_bits=$(( ($bytes_per_line*8+$tag_bits)*$associativity*$sets ))
    echo "Total size per SIMT core (bits)="$total_bits
    echo 
}

l2_cache_bits_and_size_calculations () {
    cache_name=$1
    cache_config_id=$2
    regex_l2d="^$cache_config_id[[:space:]]*[[:alpha:]]?:?([0-9]+):([0-9]+):([0-9]+),"
    regex_num_mem_controllers="^-gpgpu_n_mem[[:space:]]*([0-9]+)"
    regex_num_sub_partitions="^-gpgpu_n_sub_partition_per_mchannel[[:space:]]*([0-9]+)"

    # Find L2D config in gpgpusim.config w/ gawk
    parsed_regex=$(cat $CONFIG_FILE | gawk -v pat=$regex_l2d 'match($0, pat, a) {print a[1], a[2], a[3]}')
    parsed_arr=(${parsed_regex// / }) # Split with spaces
    sets=${parsed_arr[0]}
    bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($sets)/l(2)") )

    bytes_per_line=${parsed_arr[1]}
    bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($bytes_per_line)/l(2)") )
    bits_for_word_offset=$( printf "%.0f" $(bc -l <<< "l($bytes_per_line/($WORD_SIZE/8))/l(2)") )

    associativity=${parsed_arr[2]}

    echo "$cache_name: sets="$sets", bytes per line="$bytes_per_line", associativity="$associativity

    tag_bits=$(($ADDRESS_WIDTH-$bits_for_byte_offset-$bits_for_sets_indexing))
    echo "Bits for tag="$tag_bits
    echo "Bits for index="$bits_for_sets_indexing
    echo "Bits for byte offset="$bits_for_byte_offset
    echo "Bits for word offset="$bits_for_word_offset
    echo "Bits for tag+index="$(($tag_bits+$bits_for_sets_indexing))

    num_mem_controllers=$(cat $CONFIG_FILE | gawk -v pat=$regex_num_mem_controllers 'match($0, pat, a) {print a[1]}')
    num_sub_partitions=$(cat $CONFIG_FILE | gawk -v pat=$regex_num_sub_partitions 'match($0, pat, a) {print a[1]}')

    # There's one L2D cache per memory subpartition, each one having the 
    # configuration given by the gpgpu_cache:dl2 option.
    # Having calculated the tag bits, we can calculate the total L2 cache
    # size in bits as follows:
    total_bits=$(( ($bytes_per_line*8+$tag_bits)*$associativity*$sets*$num_mem_controllers*$num_sub_partitions ))
    echo "Total size (bits)="$total_bits
    echo
}

l1_cache_bits_and_size_calculations "L1I" "-gpgpu_cache:il1" 
l1_cache_bits_and_size_calculations "L1D" "-gpgpu_cache:dl1" 
l1_cache_bits_and_size_calculations "L1T" "-gpgpu_tex_cache:l1" 
l1_cache_bits_and_size_calculations "L1C" "-gpgpu_const_cache:l1" 
l2_cache_bits_and_size_calculations "L2D" "-gpgpu_cache:dl2" 

