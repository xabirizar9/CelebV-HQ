#!/usr/bin/env python3

import os
import torch
import tqdm
import csv
import numpy as np
import json
import warnings
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DistributedSampler, Subset
import torchaudio

# Suppress NCCL warnings
os.environ["NCCL_DEBUG"] = "WARN"
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

# Configuration
AUDIO_DIR = "data/CelebV-HQ/audio"
OUTPUT_FILE = "data/CelebV-HQ/transcripts/transcriptions_with_timestamps.txt"
TEMP_DIR = "data/CelebV-HQ/transcripts/temp"
WHISPER_MODEL = "openai/whisper-small.en"
BATCH_SIZE = 8
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = 100  # Limit to 100 samples

class AudioDataset(Dataset):
    def __init__(self, audio_dir, device):
        self.audio_dir = audio_dir
        self.device = device
        self.file_paths = []
        
        # Find all audio files
        self.file_paths.extend(list(Path(audio_dir).rglob("*.wav")))
        
        # Sort to ensure consistent ordering across processes
        self.file_paths.sort()
        
        print(f"Found {len(self.file_paths)} total audio files")
            
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = str(self.file_paths[idx])
        
        try:
            # Load and resample audio
            speech, sr = torchaudio.load(path)
            
            # Handle multi-channel audio (take first channel)
            if speech.shape[0] > 1:
                speech = speech[0].unsqueeze(0)
            
            # Resample to 16kHz if needed
            if sr != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                speech = resampler(speech)
            
            # Return the processed audio and path
            return {
                "path": path,
                "audio": speech.squeeze(0).numpy().astype(np.float32),
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a zero-length audio sample as fallback
            return {
                "path": path,
                "audio": np.zeros(0, dtype=np.float32),
            }

def collate_fn(batch):
    # Filter out failed samples
    valid_batch = [item for item in batch if len(item["audio"]) > 0]
    
    if not valid_batch:
        return {"paths": [], "audio": []}
    
    # Separate paths and audio
    paths = [item["path"] for item in valid_batch]
    audio = [item["audio"] for item in valid_batch]
    
    return {
        "paths": paths,
        "audio": audio
    }

def main():
    # Initialize distributed environment with explicit GPU mapping
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Explicitly set the CUDA device before initializing distributed
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Initialize accelerator with simplified parameters
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision="fp16"
    )
    
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    process_index = accelerator.process_index
    
    print(f"Using {accelerator.num_processes} processes on {device} (process_index: {process_index})")
    
    # Create output and temp directories
    if is_main_process:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Ensure all processes see the directories
    accelerator.wait_for_everyone()
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=torch.float16
    ).to(device)
    
    # Create full dataset
    full_dataset = AudioDataset(AUDIO_DIR, device)
    
    # Limit to MAX_SAMPLES by creating a subset (use the first MAX_SAMPLES files)
    dataset = full_dataset

    # Distribute dataset across GPUs
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False
    )
    
    # Create dataloader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Create progress bar
    pbar = tqdm.tqdm(
        total=len(dataloader), 
        desc=f"GPU {accelerator.process_index}", 
        position=accelerator.process_index
    )
    
    # Process files
    results = []
    
    for batch_idx, batch in enumerate(dataloader):
        paths = batch["paths"]
        audio = batch["audio"]
        
        if not paths:
            pbar.update(1)
            continue
            
        try:
            # Process the batch using the parameters from the example
            inputs = processor(
                audio, 
                return_tensors="pt", 
                padding="longest",
                return_attention_mask=True, 
                sampling_rate=TARGET_SAMPLE_RATE,
                truncation=False
            )
            
            # Move to device with float16
            inputs = {k: v.to(device, torch.float16) for k, v in inputs.items()}
            
            # Generate transcriptions using temperature fallback and other parameters
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    condition_on_prev_tokens=False,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=1.35,
                    return_timestamps=True
                )
            
            # Decode the generated tokens
            transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Store results
            for path, transcript in zip(paths, transcripts):
                rel_path = os.path.relpath(path, AUDIO_DIR)
                results.append((rel_path, transcript))
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            for path in paths:
                rel_path = os.path.relpath(path, AUDIO_DIR)
                results.append((rel_path, f"ERROR: {str(e)}"))
        
        pbar.update(1)
    
    pbar.close()
    
    # Save partial results from each process to avoid deadlocks in all_gather
    temp_file = os.path.join(TEMP_DIR, f"results_rank_{process_index}.json")
    with open(temp_file, 'w') as f:
        json.dump(results, f)
    
    # Make sure all processes have written their results before proceeding
    accelerator.wait_for_everyone()
    
    # Only the main process combines results and writes the final output
    if is_main_process:
        combined_results = []
        
        # Read all temporary files
        for rank in range(accelerator.num_processes):
            rank_file = os.path.join(TEMP_DIR, f"results_rank_{rank}.json")
            if os.path.exists(rank_file):
                with open(rank_file, 'r') as f:
                    try:
                        rank_results = json.load(f)
                        combined_results.extend(rank_results)
                        print(f"Loaded {len(rank_results)} results from rank {rank}")
                    except json.JSONDecodeError:
                        print(f"Error loading results from rank {rank}")
        
        print(f"Total combined results: {len(combined_results)}")
        
        # Write to output file
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["file", "transcript_with_timestamps"])  # Header
            for rel_path, transcript in combined_results:
                writer.writerow([rel_path, f'"{transcript}"'])
        
        print(f"All results saved to {OUTPUT_FILE}")
        
        # Cleanup temp files
        for rank in range(accelerator.num_processes):
            rank_file = os.path.join(TEMP_DIR, f"results_rank_{rank}.json")
            if os.path.exists(rank_file):
                os.remove(rank_file)
    
    # Make sure all processes finish cleanly
    accelerator.wait_for_everyone()
    print(f"Process {process_index} completed successfully")

if __name__ == "__main__":
    main() 