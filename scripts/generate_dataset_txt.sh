#!/bin/bash

# Check if the main path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <main_path>"
    exit 1
fi

# Define the main path from the argument
MAIN_PATH="$1"

# Define the output path
OUTPUT_FOLDER="./data"
mkdir -p "${OUTPUT_FOLDER}"

# Function to check if a file is an image
check_is_image_file() {
    local file=$1
    case "${file,,}" in
        *.png|*.jpg|*.jpeg|*.bmp|*.tiff|*.tif) return 0 ;;
        *) return 1 ;;
    esac
}

# Function to gather image files from a directory
gather_image_files() {
    local dir=$1
    find "$dir" -type f | while read -r file; do
        check_is_image_file "$file" && echo "$file"
    done | sort
}

# Function to write file paths to an output file
write_to_file() {
    local output_file=$1
    shift
    : > "$output_file" # Clear file content if it exists
    for file in "$@"; do
        echo "$file" >> "$output_file"
    done
}

# Initialize associative arrays
declare -A dataset_folder_name
declare -A output_file_name
declare -a phases=("train" "val")

# Write files to output
for phase in "${phases[@]}"; do

    # Gather image files
    gt_files=($(gather_image_files "$MAIN_PATH/${phase}/GT"))
    input_files=($(gather_image_files "$MAIN_PATH/${phase}/input"))

    # Write to txt
    write_to_file "${OUTPUT_FOLDER}/LSUI_${phase}_target.txt" "${gt_files[@]}"
    write_to_file "${OUTPUT_FOLDER}/LSUI_${phase}_input.txt" "${input_files[@]}"
done
