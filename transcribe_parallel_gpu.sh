#!/bin/bash

# Parallel script to transcribe WAV files to text using Whisper with GPU acceleration
# Optimized for multi-GPU systems

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

# Check if ffmpeg is installed (required by Whisper)
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed (required by Whisper).${NC}"
    echo -e "${YELLOW}Please install ffmpeg using:${NC}"
    echo -e "${YELLOW}sudo apt-get update && sudo apt-get install -y ffmpeg${NC}"
    exit 1
fi

# Check for GPU and PyTorch installation
echo -e "${YELLOW}Checking for GPU support...${NC}"
cat > /tmp/check_gpu.py << 'EOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            print(f"  - {torch.cuda.get_device_name(i)}")
        print("OK:GPU")
        sys.exit(0)
    else:
        print("No CUDA-capable GPU found.")
        print("OK:CPU")
        sys.exit(1)
except ImportError:
    print("PyTorch not installed. Installing PyTorch with CUDA support...")
    print("INSTALL:TORCH")
    sys.exit(2)
EOF

gpu_check=$(python3 /tmp/check_gpu.py)
gpu_status=$?

if [[ "$gpu_check" == *"INSTALL:TORCH"* ]]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install PyTorch. Falling back to CPU mode.${NC}"
        USE_GPU=0
        NUM_JOBS=56  # Use more CPU cores since no GPU
    else
        gpu_check=$(python3 /tmp/check_gpu.py)
        gpu_status=$?
    fi
fi

if [[ "$gpu_check" == *"OK:GPU"* ]] || [ $gpu_status -eq 0 ]; then
    echo -e "${GREEN}GPU support detected!${NC}"
    USE_GPU=1
    GPU_COUNT=$(echo "$gpu_check" | grep "Found" | sed 's/Found \([0-9]*\) GPU.*/\1/')
    NUM_JOBS=$GPU_COUNT  # One job per GPU
    echo -e "${GREEN}Will use $NUM_JOBS GPU(s) for processing.${NC}"
else
    echo -e "${YELLOW}No GPU detected. Using CPU mode.${NC}"
    USE_GPU=0
    NUM_JOBS=56  # Use multiple CPU cores
    echo -e "${YELLOW}Will use $NUM_JOBS CPU cores for processing.${NC}"
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

# Create the output and log directories if they don't exist
mkdir -p "$TRANSCRIPT_DIR"
mkdir -p "$LOG_DIR"

# Count total number of WAV files
echo -e "${YELLOW}Finding all WAV files (this may take a moment)...${NC}"
find "$WAV_DIR" -name "*.wav" > "$TEMP_FILE_LIST"
total_files=$(wc -l < "$TEMP_FILE_LIST")
echo -e "${GREEN}Found $total_files WAV files to process.${NC}"

# Create batches of files for processing
BATCH_SIZE=100
NUM_BATCHES=$(( (total_files + BATCH_SIZE - 1) / BATCH_SIZE ))
mkdir -p /tmp/wav_batches
split -l $BATCH_SIZE "$TEMP_FILE_LIST" /tmp/wav_batches/batch_

# Python script for transcribing with Whisper on GPU
cat > /tmp/whisper_transcribe_gpu.py << 'EOF'
import sys
import whisper
import os
import json
import torch
import time
from glob import glob

def transcribe_audio(audio_file, model_name="small", device="cuda" if torch.cuda.is_available() else "cpu"):
    try:
        # Load the model on the specified device
        model = whisper.load_model(model_name, device=device)
        
        # Transcribe audio
        result = model.transcribe(audio_file)
        return result
    except Exception as e:
        print(f"Error transcribing {audio_file}: {str(e)}")
        return None

def process_batch(batch_file, output_dir, model_name, gpu_id=None):
    # Set the GPU device if specified
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Read the batch file
    with open(batch_file, 'r') as f:
        audio_files = [line.strip() for line in f]
    
    total = len(audio_files)
    success_count = 0
    failed_count = 0
    
    start_time = time.time()
    print(f"Starting batch {os.path.basename(batch_file)} with {total} files on {device}...")
    
    # Process each file in the batch
    for i, audio_file in enumerate(audio_files):
        # Get the relative path for the output file
        rel_path = os.path.relpath(audio_file, os.environ["WAV_DIR"])
        output_file = os.path.join(output_dir, rel_path.replace('.wav', '.json'))
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Transcribe the audio
        result = transcribe_audio(audio_file, model_name, device)
        
        if result:
            # Write the result to the output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Log success
            with open(os.path.join(os.environ["LOG_DIR"], "success.txt"), 'a') as f:
                f.write(f"{rel_path}\n")
            
            success_count += 1
            print(f"[{i+1}/{total}] SUCCESS: {rel_path}")
        else:
            # Log failure
            with open(os.path.join(os.environ["LOG_DIR"], "failed.txt"), 'a') as f:
                f.write(f"{rel_path}\n")
            
            failed_count += 1
            print(f"[{i+1}/{total}] FAILED: {rel_path}")
    
    end_time = time.time()
    duration = end_time - start_time
    rate = total / duration if duration > 0 else 0
    
    print(f"Batch {os.path.basename(batch_file)} completed:")
    print(f"  - Processed: {total} files")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Time: {duration:.2f} seconds")
    print(f"  - Rate: {rate:.2f} files/second")
    
    return success_count, failed_count

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python whisper_transcribe_gpu.py <batch_file> <output_dir> <model_name> [gpu_id]")
        sys.exit(1)
    
    batch_file = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]
    gpu_id = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    success, failed = process_batch(batch_file, output_dir, model_name, gpu_id)
    print(f"SUCCESS:{success}")
    print(f"FAILED:{failed}")
EOF

# Clear previous log files
rm -f "$LOG_DIR/success.txt" "$LOG_DIR/failed.txt"

echo -e "${GREEN}Starting transcription...${NC}"
echo -e "${YELLOW}Using Whisper model: $WHISPER_MODEL${NC}"
echo -e "${YELLOW}Processing $total_files files in $NUM_BATCHES batches${NC}"
start_time=$(date +%s)

export WAV_DIR TRANSCRIPT_DIR LOG_DIR WHISPER_MODEL

# Process batches
success_total=0
failed_total=0
completed_batches=0

process_batch() {
    local batch_file=$1
    local gpu_id=$2
    
    # Run the Python script to process the batch
    local output=$(python3 /tmp/whisper_transcribe_gpu.py "$batch_file" "$TRANSCRIPT_DIR" "$WHISPER_MODEL" "$gpu_id")
    
    # Extract success and failed counts
    local success=$(echo "$output" | grep "SUCCESS:" | cut -d':' -f2)
    local failed=$(echo "$output" | grep "FAILED:" | cut -d':' -f2)
    
    echo "$success:$failed"
}

if [ $USE_GPU -eq 1 ]; then
    # GPU mode: Process batches in parallel, one batch per GPU
    echo -e "${GREEN}Processing in GPU mode with $NUM_JOBS GPU(s)${NC}"
    
    batch_files=($(ls -1 /tmp/wav_batches/batch_*))
    
    # Process batches in groups based on GPU count
    for ((i=0; i<${#batch_files[@]}; i+=$NUM_JOBS)); do
        batch_group=("${batch_files[@]:i:NUM_JOBS}")
        running_jobs=()
        results=()
        
        # Start a job for each batch in the group
        for ((j=0; j<${#batch_group[@]}; j++)); do
            if [ $j -lt $NUM_JOBS ]; then
                echo -e "${YELLOW}Starting batch $(($i+$j+1))/$NUM_BATCHES on GPU $j${NC}"
                process_batch "${batch_group[$j]}" $j > /tmp/batch_result_$j.txt &
                running_jobs+=($!)
            fi
        done
        
        # Wait for all jobs in this group to finish
        for ((j=0; j<${#running_jobs[@]}; j++)); do
            wait ${running_jobs[$j]}
            
            # Read results
            if [ -f /tmp/batch_result_$j.txt ]; then
                result=$(cat /tmp/batch_result_$j.txt)
                success=$(echo $result | cut -d':' -f1)
                failed=$(echo $result | cut -d':' -f2)
                
                success_total=$((success_total + success))
                failed_total=$((failed_total + failed))
                completed_batches=$((completed_batches + 1))
                
                # Progress update
                progress=$((completed_batches * 100 / NUM_BATCHES))
                echo -e "${GREEN}[$progress%] Completed: $completed_batches/$NUM_BATCHES batches${NC}"
                
                rm /tmp/batch_result_$j.txt
            fi
        done
    done
else
    # CPU mode: Process batches sequentially
    echo -e "${GREEN}Processing in CPU mode${NC}"
    
    for batch_file in /tmp/wav_batches/batch_*; do
        echo -e "${YELLOW}Starting batch $((completed_batches+1))/$NUM_BATCHES${NC}"
        
        result=$(process_batch "$batch_file" "")
        success=$(echo $result | cut -d':' -f1)
        failed=$(echo $result | cut -d':' -f2)
        
        success_total=$((success_total + success))
        failed_total=$((failed_total + failed))
        completed_batches=$((completed_batches + 1))
        
        # Progress update
        progress=$((completed_batches * 100 / NUM_BATCHES))
        echo -e "${GREEN}[$progress%] Completed: $completed_batches/$NUM_BATCHES batches${NC}"
    done
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
processed=$((success_total + failed_total))

# Print summary
echo -e "\n${GREEN}Transcription Summary:${NC}"
echo -e "${GREEN}Total processed: $processed of $total_files${NC}"
echo -e "${GREEN}Successfully transcribed: $success_total${NC}"
echo -e "${RED}Failed transcriptions: $failed_total${NC}"
echo -e "${GREEN}Time taken: $hours hours, $minutes minutes, $seconds seconds${NC}"
echo -e "${GREEN}Processing speed: $(echo "scale=2; $processed / $duration" | bc) files per second${NC}"

if [ $failed_total -gt 0 ]; then
    echo -e "${RED}Failed transcriptions are logged in: $LOG_DIR/failed.txt${NC}"
fi

# Clean up
rm -f "$TEMP_FILE_LIST" "/tmp/whisper_transcribe_gpu.py" "/tmp/check_gpu.py"
rm -rf /tmp/wav_batches

echo -e "${GREEN}Transcription completed!${NC}"
echo -e "${YELLOW}Transcriptions are saved in: $TRANSCRIPT_DIR${NC}" 