#!/bin/bash
set -e
# This file calculates the total size (in bits) of L1D, L1T, L1C and L2 cache.
# Accepts a filepath to a gpgpusim.config file as argument. If none, 
# it tries to find it inside the current dir.
ADDRESS_WIDTH=64
CONFIG_FILE=./gpgpusim.config
if [[ ! -z $1 ]] 
then
    if test -f $1; then
        CONFIG_FILE=$1
    fi;
fi;

# Calculates total cache size for a specific cache, given the cache config. 
# $1: The cache name (e.g. L1D). Will also be used to export the appropriate var, i.e. L1D_SIZE_BITS
# $2: The name of the variable to read from the gpgpusim.config file. E.g. for L1D it's "-gpgpu_cache:il1".
cache_bits_and_size_calculations () {
    cache_name=$1
    cache_config_id=$2
    # L2 is the only cache type with subpartitions
    [[ "${cache_name:1:1}" = "2" ]] && has_sub_partitions=1 || has_sub_partitions=0

    cache_config_regex="^$cache_config_id[[:space:]]*([[:alpha:]])?:?([0-9]+):([0-9]+):([0-9]+),"
    regex_num_mem_controllers="^-gpgpu_n_mem[[:space:]]*([0-9]+)"
    regex_num_sub_partitions="^-gpgpu_n_sub_partition_per_mchannel[[:space:]]*([0-9]+)"

    # Find cache config in gpgpusim.config w/ gawk
    parsed_regex=$(cat $CONFIG_FILE | gawk -v pat=$cache_config_regex 'match($0, pat, a) {print a[1], a[2], a[3], a[4]}')
    parsed_arr=(${parsed_regex// / }) # Split with spaces.
    [[ ${parsed_arr[0]} = "S" ]] && is_sectored=1 || is_sectored=0 # Check if sectored.

    sets=${parsed_arr[1]}
    # Log2 of number of sets = num bits for indexing
    bits_for_sets_indexing=$( printf "%.0f" $(bc -l <<< "l($sets)/l(2)") ) 

    bytes_per_line=${parsed_arr[2]}
    # Log2 of number of bytes per line = bits for byte indexing 
    bits_for_byte_offset=$( printf "%.0f" $(bc -l <<< "l($bytes_per_line)/l(2)") )

    associativity=${parsed_arr[3]}

    echo -n "$cache_name: sets="$sets", bytes per line="$bytes_per_line", associativity="$associativity
    if [[ $is_sectored -ne 0 ]]; then
        echo " (Sectored)"
    else
        echo ""
    fi 
    tag_bits=$(($ADDRESS_WIDTH-$bits_for_byte_offset-$bits_for_sets_indexing))
    echo "Bits for tag="$tag_bits
    echo "Bits for index="$bits_for_sets_indexing
    echo "Bits for byte offset="$bits_for_byte_offset
    echo "Bits for tag+index="$(($tag_bits+$bits_for_sets_indexing))
    
    # Size calculation, not taking into account caches which are present in multiple
    # memory controllers (i.e. L2) 
    if [[ $is_sectored -ne 0 ]]; then
        num_sectors=$(cat src/abstract_hardware_model.h | gawk -v pat="SECTOR_CHUNCK_SIZE[[:space:]]*=[[:space:]]*([0-9]+);" 'match($0, pat, a) {print a[1]}')
        echo "Number of sectors=$num_sectors"
        total_bits=$(( ($bytes_per_line*8)*$associativity*$sets ))
        # Assume one tag field per sector, i.e. divide total cache lines by sector size.
        total_bits=$(( $total_bits + $tag_bits*(($associativity*$sets)/$num_sectors) ))
    else # Non-sectored
        total_bits=$(( ($bytes_per_line*8+$tag_bits)*$associativity*$sets ))
    fi

    if [[ $has_sub_partitions -ne 0 ]]; then
        num_mem_controllers=$(cat $CONFIG_FILE | gawk -v pat=$regex_num_mem_controllers 'match($0, pat, a) {print a[1]}')
        num_sub_partitions=$(cat $CONFIG_FILE | gawk -v pat=$regex_num_sub_partitions 'match($0, pat, a) {print a[1]}')

        echo "Number of mem controllers="$num_mem_controllers
        echo "Number of partitions per controller="$num_sub_partitions
        # There's one cache instance per memory subpartition, each one having the 
        # configuration given by the gpgpu_cache option in gpgpusim.config.
        # Having calculated the total bits, we can calculate the total cache
        # size by multiplying with the number of mem controllers and the number
        # of sub partitions that each controller has.
        total_bits=$(( $total_bits*$num_mem_controllers*$num_sub_partitions ))
        echo "Total size (bits)="$total_bits
    else
        echo "Total size per SIMT core (bits)="$total_bits
    fi
    echo
    export $1"_SIZE_BITS"=$total_bits

}

cache_bits_and_size_calculations "L1I" "-gpgpu_cache:il1" 
cache_bits_and_size_calculations "L1D" "-gpgpu_cache:dl1" 
cache_bits_and_size_calculations "L1T" "-gpgpu_tex_cache:l1"
cache_bits_and_size_calculations "L1C" "-gpgpu_const_cache:l1"
cache_bits_and_size_calculations "L2" "-gpgpu_cache:dl2"

