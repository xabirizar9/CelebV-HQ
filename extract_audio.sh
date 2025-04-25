#!/bin/bash

# Script to extract audio from all MP4 files in the CelebV-HQ directory
# and save them as WAV files with the same names

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source directory containing MP4 files
MP4_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/35666"
# Output directory for WAV files (will follow the same structure)
WAV_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/audio"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed.${NC}"
    echo -e "${YELLOW}Please install ffmpeg using:${NC}"
    echo -e "${YELLOW}sudo apt-get update && sudo apt-get install -y ffmpeg${NC}"
    exit 1
fi

# Check if the source directory exists
if [ ! -d "$MP4_DIR" ]; then
    echo -e "${RED}Error: Source directory $MP4_DIR does not exist.${NC}"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$WAV_DIR"

# Count total number of MP4 files
total_files=$(find "$MP4_DIR" -name "*.mp4" | wc -l)
echo -e "${GREEN}Found $total_files MP4 files to process.${NC}"

# Initialize counters
processed=0
success=0
failed=0
no_audio=0

# Process each MP4 file
find "$MP4_DIR" -name "*.mp4" | while read mp4_file; do
    # Get the relative path of the file
    rel_path=$(realpath --relative-to="$MP4_DIR" "$mp4_file")
    # Create output directory structure
    output_dir=$(dirname "$WAV_DIR/$rel_path")
    mkdir -p "$output_dir"
    
    # Create WAV filename by replacing .mp4 with .wav
    wav_file="${mp4_file%.mp4}.wav"
    wav_file="$WAV_DIR/${wav_file#$MP4_DIR/}"
    
    # Update progress
    processed=$((processed+1))
    percent=$((processed*100/total_files))
    echo -e "${YELLOW}[$percent%] Processing ($processed/$total_files): $rel_path${NC}"
    
    # Check if the video has an audio stream
    has_audio=$(ffprobe -i "$mp4_file" -show_streams -select_streams a -loglevel error 2>/dev/null)
    
    if [ -n "$has_audio" ]; then
        # Extract audio to WAV file
        if ffmpeg -i "$mp4_file" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$wav_file" -y -loglevel error; then
            echo -e "${GREEN}Success: $wav_file${NC}"
            success=$((success+1))
        else
            echo -e "${RED}Failed: $wav_file${NC}"
            failed=$((failed+1))
        fi
    else
        echo -e "${YELLOW}No audio stream found in: $rel_path${NC}"
        # Create an empty log entry for files with no audio
        echo "$rel_path" >> "$WAV_DIR/no_audio_files.log"
        no_audio=$((no_audio+1))
    fi
done

# Print summary
echo -e "\n${GREEN}Audio Extraction Summary:${NC}"
echo -e "${GREEN}Total processed: $processed${NC}"
echo -e "${GREEN}Successfully extracted: $success${NC}"
echo -e "${YELLOW}Files with no audio: $no_audio${NC}"
echo -e "${RED}Failed extractions: $failed${NC}"

if [ $no_audio -gt 0 ]; then
    echo -e "${YELLOW}Files with no audio are logged in: $WAV_DIR/no_audio_files.log${NC}"
fi

echo -e "${GREEN}Audio extraction completed!${NC}" 