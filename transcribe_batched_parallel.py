#!/usr/bin/env python3

import os
import torch
import tqdm
import csv
import torchaudio
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DistributedSampler

# Configuration
AUDIO_DIR = "data/CelebV-HQ/audio"
OUTPUT_FILE = "data/CelebV-HQ/transcripts/transcriptions.txt"
WHISPER_MODEL = "openai/whisper-small"
BATCH_SIZE = 32
MAX_NEW_TOKENS = 444
TARGET_SAMPLE_RATE = 16000

class AudioDataset(Dataset):
    def __init__(self, audio_dir, processor, device, processed_files=None):
        self.audio_dir = audio_dir
        self.processor = processor
        self.device = device
        self.file_paths = []
        
        # Find all audio files
        self.file_paths.extend(list(Path(audio_dir).rglob("*.wav")))
        
        # Filter out already processed files
        if processed_files:
            self.file_paths = [f for f in self.file_paths if str(f) not in processed_files]
            
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
                "audio": speech.squeeze(0).numpy(),
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a zero-length audio sample as fallback
            return {
                "path": path,
                "audio": torch.zeros(0).numpy(),
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
    # Set CUDA device explicitly based on local rank before initializing accelerator
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Initialize accelerator for distributed processing - let it handle device mapping
    accelerator = Accelerator(device_placement=True)
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    
    print(f"Using {accelerator.num_processes} processes on {device} (local_rank: {local_rank})")
    if is_main_process:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    
    # Configure model to avoid forced_decoder_ids conflict
    model_config = WhisperForConditionalGeneration.config_class.from_pretrained(WHISPER_MODEL)
    model_config.forced_decoder_ids = None
    model_config.suppress_tokens = []
    
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL, 
        config=model_config,
        torch_dtype=torch.float16
    ).to(device)
    
    # Load checkpoint (only on main process)
    processed_files = set()
    
    # Create dataset and distribute across GPUs
    dataset = AudioDataset(AUDIO_DIR, processor, device, processed_files if is_main_process else None)
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
        num_workers=16,
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
            # Process the entire batch at once
            inputs = processor(
                audio, 
                return_tensors="pt", 
                padding="max_length",
                return_attention_mask=True, 
                sampling_rate=TARGET_SAMPLE_RATE,
                feature_size=80,  # Default feature dim for Whisper
                max_length=3000   # Fixed length for Whisper
            )
            
            # Make sure input_features have the right shape
            if 'input_features' in inputs and inputs['input_features'].shape[-1] != 3000:
                print(f"Warning: Input features shape before padding: {inputs['input_features'].shape}")
                # Need to pad to exactly 3000 in time dimension
                if inputs['input_features'].shape[-1] < 3000:
                    padding_size = 3000 - inputs['input_features'].shape[-1]
                    # Pad the last dimension (time) to 3000
                    inputs['input_features'] = torch.nn.functional.pad(
                        inputs['input_features'], 
                        (0, padding_size)
                    )
                else:
                    # Truncate if longer
                    inputs['input_features'] = inputs['input_features'][..., :3000]
                
                print(f"Input features shape after fixing: {inputs['input_features'].shape}")
            
            # Move to device
            inputs = {k: v.to(device, torch.float16) for k, v in inputs.items()}
            
            # Debug dimensions for first few batches
            if batch_idx < 2:
                print(f"Input features shape: {inputs['input_features'].shape}")
                if 'attention_mask' in inputs:
                    print(f"Attention mask shape: {inputs['attention_mask'].shape}")
            
            # Generate transcriptions for the entire batch
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    decoder_input_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe")
                )
            
            # Decode all transcripts in the batch
            transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Store all results
            for path, transcript in zip(paths, transcripts):
                rel_path = os.path.relpath(path, AUDIO_DIR)
                results.append((rel_path, transcript))
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Gather and save results
    accelerator.wait_for_everyone()
    
    # Use try-finally to ensure proper process group cleanup
    try:
        if is_main_process:
            # Fix: Use proper all-gather to collect results from all processes
            gathered_results = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(gathered_results, results)
            
            # Flatten results properly
            flat_results = []
            for result_list in gathered_results:
                if result_list:  # Check if not None
                    flat_results.extend(result_list)
            
            print(f"Collected {len(flat_results)} transcriptions from all GPUs")
            
            # Write to output file
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for rel_path, transcript in flat_results:
                    writer.writerow([rel_path, f'"{transcript}"'])
            
            print(f"All results saved to {OUTPUT_FILE}")
            print(f"Total transcriptions: {len(flat_results)}")
    finally:
        # Force cleanup of NCCL/distributed resources
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()