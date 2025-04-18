from typing import Callable, Optional, Tuple, List
import os
import numpy as np
import torch
import os
import glob
import torchaudio
from torch.utils.data import Dataset
from hyperparams import TVAEParams


class TVAESeqSeqDataset(Dataset):
    def __init__(self, hp: TVAEParams) -> None:
        self.target_sr = hp.sampling_rate
        self.seq_length = hp.seq_len
        self.step = hp.window_step
        
        self.data = []  # List of tuples: (waveform, genre, filename)
        self.index = []  # List of tuples: (data_index, start_sample)
        
        # Get selected genres
        selected_genres = hp.selected_genre if isinstance(hp.selected_genre, list) else [hp.selected_genre]
        # Traverse genre folders
        try:
            genre_dirs = [
                d for d in os.listdir(hp.data_genres_original_dir)
                if os.path.isdir(os.path.join(hp.data_genres_original_dir, d)) and d in selected_genres
            ]
            print(f"Filtered {len(genre_dirs)} genres: {genre_dirs}")
        except Exception as e:
            print(f"Error listing directories: {e}")
            genre_dirs = []
            
        for genre in genre_dirs:
            genre_path = os.path.join(hp.data_genres_original_dir, genre)
            wav_files = glob.glob(os.path.join(genre_path, "*.wav"))
            
            for wav_file in wav_files:
                try:
                    waveform, sr = torchaudio.load(wav_file)
                except Exception as e:
                    print(f"Error loading {wav_file}: {e}")
                    continue
                
                # Resample if needed
                if sr != self.target_sr:
                    try:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                        waveform = resampler(waveform)
                    except Exception as e:
                        print(f"Resample error {wav_file}: {e}")
                        continue
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Skip files that are too short for at least one window
                if waveform.shape[1] < self.seq_length:
                    continue
                
                # Keep full audio length (no truncation)
                data_idx = len(self.data)
                self.data.append((waveform.contiguous(), genre, os.path.basename(wav_file)))
                
                # Create sliding windows across full audio length
                num_samples = waveform.shape[1]
                for start in range(0, num_samples - self.seq_length, self.step):
                    self.index.append((data_idx, start))
                    
        print(f"Loaded {len(self.data)} files with {len(self.index)} segments")

    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_idx, start = self.index[idx]
        waveform, genre, filename = self.data[data_idx]
        
        # Extract src: [start, start+seq_length]
        src = waveform[:, start:start + self.seq_length]
        # Extract tgt: [start+1, start+seq_length+1]
        tgt = waveform[:, start + 1:start + self.seq_length + 1]
        
        return src, tgt