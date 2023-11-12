#!/bin/bash
# Script for replacing a SASS instruction in a CUDA executable.

# The full path to the CUDA executable to inject
app_binary_path=$1
original_instruction_hex=$2
injected_instruction_hex=$3

# The kernel name will be used to limit the search in the
# executable. If none supplied, the whole file will be searched
# and replaced.
kernel_name=$4

if [ -z "$app_binary_path" ]; then
    echo "gpuFI: ERROR: A path to a cuda executable must be supplied."
    exit 1
fi
if [ ! -f "$app_binary_path" ]; then
    echo "gpuFI: ERROR: No such file $app_binary_path"
    exit 1
fi
if [ -z "$original_instruction_hex" ] || [ -z "$injected_instruction_hex" ]; then
    echo "gpuFI: ERROR: Please supply both the original instruction and the instruction to replace it with in hex format"
    exit 1
fi

# Make sure the supplied file is an ELF
read -r -n 4 filetype <"$app_binary_path"

if ! echo "$filetype" | grep "ELF" 1>/dev/null; then
    echo "gpuFI: ERROR: File is not an ELF executable"
    exit 1
fi

# Make sure the kernel supplied exists in the executable
if [ -n "$kernel_name" ] && ! grep "nv.info.$kernel_name" "$app_binary_path" 1>/dev/null; then
    echo "gpuFI: ERROR: \"nv.info.$kernel_name\" header was not found in the supplied executable"
    exit 1
fi

# Find where the kernel starts
# gpuFI TODO: Find the end of the kernel to limit the search even more.
kernel_byte_offset_in_file=$(grep --byte-offset --only-matching --text "nv.info.$kernel_name" "$app_binary_path" | cut -d':' -f 1)

full_binary_dump=$(xxd -c 0 -p "$app_binary_path")
# Only search after the offset we found before. I could
# not find a more elegant way, so I split the file at the offset,
# manipulate it with sed and then put it back together.
# Note that the *2 are added due to each byte being in hex.
new_file_first_part=$(echo "$full_binary_dump" | head -c $(((kernel_byte_offset_in_file - 1) * 2)))
new_file_last_part=$(echo "$full_binary_dump" | tail -c $((${#full_binary_dump} - ((kernel_byte_offset_in_file - 1) * 2) - 1)) | sed "s/${original_instruction_hex}/${injected_instruction_hex}/g")

# Put the file back together, using xxd.
echo "${new_file_first_part}${new_file_last_part}" | xxd -p -r >"$app_binary_path"_injected
