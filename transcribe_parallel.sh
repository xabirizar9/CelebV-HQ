#!/bin/bash

# Parallel script to transcribe WAV files to text using Whisper
# Optimized for multi-core systems

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source directory containing WAV files
WAV_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/audio"
# Output directory for transcriptions
TRANSCRIPT_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/transcripts"
# Log directory
LOG_DIR="/home/xabieririzar/CelebV-HQ/data/CelebV-HQ/transcripts/logs"
# Temporary file list
TEMP_FILE_LIST="/tmp/wav_file_list.txt"
# Number of CPU cores to use
NUM_CORES=56  # Using 56 of 60 cores to leave some resources for system
# Whisper model size: tiny, base, small, medium, large
# Smaller models are faster but less accurate
WHISPER_MODEL="small"  # Good balance between speed and accuracy

# Check if Python and pip are installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo -e "${YELLOW}Please install Python 3 using:${NC}"
    echo -e "${YELLOW}sudo apt-get update && sudo apt-get install python3 python3-pip${NC}"
    exit 1
fi

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo -e "${RED}Error: GNU parallel is not installed.${NC}"
    echo -e "${YELLOW}Please install parallel using:${NC}"
    echo -e "${YELLOW}sudo apt-get update && sudo apt-get install -y parallel${NC}"
    exit 1
fi

# Check if whisper is installed
if ! pip3 show openai-whisper &> /dev/null; then
    echo -e "${YELLOW}OpenAI Whisper is not installed. Installing now...${NC}"
    pip3 install openai-whisper
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install OpenAI Whisper. Please install it manually:${NC}"
        echo -e "${YELLOW}pip3 install openai-whisper${NC}"
        exit 1
    fi
fi

# Check if ffmpeg is installed (required by Whisper)
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed (required by Whisper).${NC}"
    echo -e "${YELLOW}Please install ffmpeg using:${NC}"
    echo -e "${YELLOW}sudo apt-get update && sudo apt-get install -y ffmpeg${NC}"
    exit 1
fi

# Create the output and log directories if they don't exist
mkdir -p "$TRANSCRIPT_DIR"
mkdir -p "$LOG_DIR"

# Count total number of WAV files
echo -e "${YELLOW}Finding all WAV files (this may take a moment)...${NC}"
find "$WAV_DIR" -name "*.wav" > "$TEMP_FILE_LIST"
total_files=$(wc -l < "$TEMP_FILE_LIST")
echo -e "${GREEN}Found $total_files WAV files to process using $NUM_CORES CPU cores.${NC}"

# Python script for transcribing with Whisper
cat > /tmp/whisper_transcribe.py << 'EOF'
import sys
import whisper
import os
import json

def transcribe_audio(audio_file, model_name="small"):
    # Load the model
    model = whisper.load_model(model_name)
    
    try:
        # Transcribe audio
        result = model.transcribe(audio_file)
        return result
    except Exception as e:
        print(f"Error transcribing {audio_file}: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python whisper_transcribe.py <audio_file> <output_file> <model_name>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    model_name = sys.argv[3]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Transcribe the audio
    result = transcribe_audio(audio_file, model_name)
    
    if result:
        # Write the result to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"SUCCESS: Transcribed {audio_file} to {output_file}")
        sys.exit(0)
    else:
        print(f"FAILED: Could not transcribe {audio_file}")
        sys.exit(1)
EOF

# Function to process a single file
process_file() {
    wav_file="$1"
    
    # Get the relative path of the file
    rel_path=$(realpath --relative-to="$WAV_DIR" "$wav_file")
    # Output filename with .json extension
    transcript_file="$TRANSCRIPT_DIR/${rel_path%.wav}.json"
    
    # Create output directory structure
    output_dir=$(dirname "$transcript_file")
    mkdir -p "$output_dir"
    
    # Prepare result files
    success_file="$LOG_DIR/success.txt"
    failed_file="$LOG_DIR/failed.txt"
    
    # Run the Python script to transcribe the audio
    if python3 /tmp/whisper_transcribe.py "$wav_file" "$transcript_file" "$WHISPER_MODEL"; then
        # Success - log to success file
        echo "$rel_path" >> "$success_file"
        echo "SUCCESS: $rel_path"
        return 0
    else
        # Failed - log to failed file
        echo "$rel_path" >> "$failed_file"
        echo "FAILED: $rel_path"
        return 1
    fi
}

export -f process_file
export WAV_DIR TRANSCRIPT_DIR LOG_DIR WHISPER_MODEL

# Clear previous log files
rm -f "$LOG_DIR/success.txt" "$LOG_DIR/failed.txt"

echo -e "${GREEN}Starting parallel transcription...${NC}"
echo -e "${YELLOW}Using Whisper model: $WHISPER_MODEL${NC}"
echo -e "${YELLOW}This process may take several hours for 30,000 files.${NC}"
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

processed=$((success + failed))

# Print summary
echo -e "\n${GREEN}Transcription Summary:${NC}"
echo -e "${GREEN}Total processed: $processed of $total_files${NC}"
echo -e "${GREEN}Successfully transcribed: $success${NC}"
echo -e "${RED}Failed transcriptions: $failed${NC}"
echo -e "${GREEN}Time taken: $hours hours, $minutes minutes, $seconds seconds${NC}"
echo -e "${GREEN}Processing speed: $(echo "scale=2; $processed / $duration" | bc) files per second${NC}"

if [ $failed -gt 0 ]; then
    echo -e "${RED}Failed transcriptions are logged in: $LOG_DIR/failed.txt${NC}"
fi

# Clean up
rm -f "$TEMP_FILE_LIST" "/tmp/whisper_transcribe.py"

echo -e "${GREEN}Parallel transcription completed!${NC}"
echo -e "${YELLOW}Transcriptions are saved in: $TRANSCRIPT_DIR${NC}"