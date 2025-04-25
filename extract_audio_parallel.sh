#!/bin/bash

# Parallel script to extract audio from all MP4 files in the CelebV-HQ directory
# and save them as WAV files with the same names
# Optimized for multi-core systems

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source directory containing MP4 files
MP4_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/35666"
# Output directory for WAV files (will follow the same structure)
WAV_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/audio"
# Log directory
LOG_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/audio/logs"
# Temporary file list
TEMP_FILE_LIST="/tmp/mp4_file_list.txt"
# Number of CPU cores to use (adjust as needed)
NUM_CORES=56  # Using 56 of 60 cores to leave some resources for system

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo -e "${RED}Error: GNU parallel is not installed.${NC}"
    echo -e "${YELLOW}Please install parallel using:${NC}"
    echo -e "${YELLOW}sudo apt-get update && sudo apt-get install -y parallel${NC}"
    exit 1
fi

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

# Create the output and log directories if they don't exist
mkdir -p "$WAV_DIR"
mkdir -p "$LOG_DIR"

# Count total number of MP4 files
echo -e "${YELLOW}Finding all MP4 files (this may take a moment)...${NC}"
find "$MP4_DIR" -name "*.mp4" > "$TEMP_FILE_LIST"
total_files=$(wc -l < "$TEMP_FILE_LIST")
echo -e "${GREEN}Found $total_files MP4 files to process using $NUM_CORES CPU cores.${NC}"

# Function to process a single file
process_file() {
    mp4_file="$1"
    
    # Get the relative path of the file
    rel_path=$(realpath --relative-to="$MP4_DIR" "$mp4_file")
    # Create output directory structure
    output_dir=$(dirname "$WAV_DIR/$rel_path")
    mkdir -p "$output_dir"
    
    # Create WAV filename by replacing .mp4 with .wav
    wav_file="${mp4_file%.mp4}.wav"
    wav_file="$WAV_DIR/${wav_file#$MP4_DIR/}"
    
    # Prepare result files
    success_file="$LOG_DIR/success.txt"
    failed_file="$LOG_DIR/failed.txt"
    no_audio_file="$LOG_DIR/no_audio_files.txt"
    
    # Check if the video has an audio stream
    has_audio=$(ffprobe -i "$mp4_file" -show_streams -select_streams a -loglevel error 2>/dev/null)
    
    if [ -n "$has_audio" ]; then
        # Extract audio to WAV file
        if ffmpeg -i "$mp4_file" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$wav_file" -y -loglevel error; then
            # Success - log to success file
            echo "$rel_path" >> "$success_file"
            echo "SUCCESS: $rel_path"
        else
            # Failed - log to failed file
            echo "$rel_path" >> "$failed_file"
            echo "FAILED: $rel_path"
        fi
    else
        # No audio - log to no_audio file
        echo "$rel_path" >> "$no_audio_file"
        echo "NO AUDIO: $rel_path"
    fi
}

export -f process_file
export MP4_DIR WAV_DIR LOG_DIR

# Clear previous log files
rm -f "$LOG_DIR/success.txt" "$LOG_DIR/failed.txt" "$LOG_DIR/no_audio_files.txt"

echo -e "${GREEN}Starting parallel processing...${NC}"
start_time=$(date +%s)

# Process files in parallel
cat "$TEMP_FILE_LIST" | parallel --progress --jobs $NUM_CORES process_file

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Count results
if [ -f "$LOG_DIR/success.txt" ]; then
    success=$(wc -l < "$LOG_DIR/success.txt")
else
    success=0
fi

if [ -f "$LOG_DIR/failed.txt" ]; then
    failed=$(wc -l < "$LOG_DIR/failed.txt")
else
    failed=0
fi

if [ -f "$LOG_DIR/no_audio_files.txt" ]; then
    no_audio=$(wc -l < "$LOG_DIR/no_audio_files.txt")
else
    no_audio=0
fi

processed=$((success + failed + no_audio))

# Print summary
echo -e "\n${GREEN}Audio Extraction Summary:${NC}"
echo -e "${GREEN}Total processed: $processed of $total_files${NC}"
echo -e "${GREEN}Successfully extracted: $success${NC}"
echo -e "${YELLOW}Files with no audio: $no_audio${NC}"
echo -e "${RED}Failed extractions: $failed${NC}"
echo -e "${GREEN}Time taken: $hours hours, $minutes minutes, $seconds seconds${NC}"
echo -e "${GREEN}Processing speed: $(echo "scale=2; $processed / $duration" | bc) files per second${NC}"

if [ $no_audio -gt 0 ]; then
    echo -e "${YELLOW}Files with no audio are logged in: $LOG_DIR/no_audio_files.txt${NC}"
fi

if [ $failed -gt 0 ]; then
    echo -e "${RED}Failed extractions are logged in: $LOG_DIR/failed.txt${NC}"
fi

# Clean up
rm -f "$TEMP_FILE_LIST"

echo -e "${GREEN}Parallel audio extraction completed!${NC}" 