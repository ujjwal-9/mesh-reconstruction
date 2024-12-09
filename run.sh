#!/bin/bash

# Point Cloud Reconstruction Wrapper Script

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
DEFAULT_METHOD="poisson"
DEFAULT_OUTPUT_DIR="outputs"

# Function to display usage instructions
usage() {
    echo -e "${YELLOW}Usage:${NC} $0 [options] <input_file>"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE       Input point cloud file (PLY format)"
    echo "  -m, --method METHOD     Reconstruction method (default: poisson)"
    echo "                         Available methods: poisson, hole_preserve"
    echo "  -o, --output DIR       Output directory (default: outputs)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo -e "${GREEN}Example:${NC}"
    echo "  $0 ../point_clouds/shoe_pc.ply"
    echo "  $0 -m hole_preserve -o custom_output ../point_clouds/shoe_pc.ply"
}

# Parse command-line arguments
METHOD=$DEFAULT_METHOD
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
INPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
                shift
            else
                echo -e "${RED}Error: Too many arguments${NC}"
                usage
                exit 1
            fi
            ;;
    esac
done

# Validate input file
if [[ -z "$INPUT_FILE" ]]; then
    echo -e "${RED}Error: No input file provided${NC}"
    usage
    exit 1
fi

# Check if input file exists and is a valid file
if [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' does not exist${NC}"
    exit 1
fi

# Validate reconstruction method
if [[ "$METHOD" != "poisson" && "$METHOD" != "hole_preserve" ]]; then
    echo -e "${RED}Error: Invalid reconstruction method '$METHOD'${NC}"
    usage
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print reconstruction details
echo -e "${GREEN}Starting Point Cloud Reconstruction${NC}"
echo -e "${YELLOW}Input File:${NC} $INPUT_FILE"
echo -e "${YELLOW}Method:${NC} $METHOD"
echo -e "${YELLOW}Output Directory:${NC} $OUTPUT_DIR"

# Run the Python script and capture output and errors if any.
python src/main.py -i "$INPUT_FILE" -m "$METHOD" -o "$OUTPUT_DIR"

# Check the exit status of the Python script.
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Reconstruction completed successfully!${NC}"
else
    echo -e "${RED}Reconstruction failed.${NC}"
    exit 1
fi