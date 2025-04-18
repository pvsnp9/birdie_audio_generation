import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from typing import  Callable, List
from hyperparams import TVAEParams
from dataclasses import asdict
from earlystopping import EarlyStopping
from utils import *
from tcrossattn_vae import TransformerCrossAttentionVAE
from birdie_utils import *
from tqdm import tqdm
import random

class BirdieTriainTVAE():
    def __init__(self, hp: TVAEParams, model:TransformerCrossAttentionVAE, optimizer: optim.Adam, criterion: Callable):
        self.hp = hp
        self.model = model
        self.optimzer = optimizer
        self.criterion = criterion
        self.es = EarlyStopping(patience=5, min_delta=0.001)

    def train(self, train_loader:DataLoader, val_loader: DataLoader):
        print("Initiating Birdie TVAE Training on", self.hp.device)
        logs: List[dict] = []

        # The Birdie tasks we'll sample from each batch
        birdie_objectives = ["infill", "deshuffle", "selective", "copy"]

        for epoch in range(self.hp.num_epochs):
            self.model.train()
            
            total_loss = 0.0
            total_mse_loss = 0.0
            total_kl_loss = 0.0
            dataset_size = len(train_loader.dataset)

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.hp.num_epochs}") as pbar:
                for idx, (src, tgt) in enumerate(train_loader):
                    src, tgt = src.to(self.hp.device), tgt.to(self.hp.device)  # src, tgt => shape [B, 1, seq_len]
                    
                    # Select a Birdie objective randomly
                    task = random.choice(birdie_objectives)

                    # Transform (src, tgt) => (birdie_src, birdie_tgt) for each objective
                    if task == "infill":
                        birdie_src, birdie_tgt = mask_middle_waveform_batch(src, mask_ratio=0.3)
                    elif task == "deshuffle":
                        chunk_size = 256  # or any multiple that divides seq_len
                        birdie_src, birdie_tgt = deshuffle_waveform_batch(src, chunk_size)
                    elif task == "selective":
                        birdie_src, birdie_tgt = selective_copy_waveform_batch(src, span_len=128)
                    else:  # "copy"
                        birdie_src, birdie_tgt = copying_waveform_batch(src)
                    

                    birdie_src, birdie_tgt = birdie_src.to(self.hp.device), birdie_tgt.to(self.hp.device)

                    self.optimzer.zero_grad()
                    # forward
                    recon_x, mu, logvar = self.model(birdie_src, birdie_tgt) # recon [B, seq, C] C is 1
                    birdie_tgt = birdie_tgt.permute(0, 2, 1) # [B, C, Seq] => [B, seq, C]
                    
                    loss, mse, kl = self.criterion(birdie_tgt, recon_x, mu, logvar)

                    loss.backward()
                    self.optimzer.step()

                    # Track sum of losses
                    batch_size = src.size(0)
                    total_loss += loss.item() * batch_size
                    total_mse_loss += mse.item() * batch_size
                    total_kl_loss += kl.item() * batch_size

                    # Update progress bar
                    current_loss = total_loss / ((idx + 1) * batch_size)
                    current_mse  = total_mse_loss / ((idx + 1) * batch_size)
                    current_kl   = total_kl_loss / ((idx + 1) * batch_size)

                    pbar.set_postfix({
                        "LOSS": f"{current_loss:.3f}",
                        "MSE":  f"{current_mse:.3f}",
                        "KL":   f"{current_kl:.3f}"
                    })

                    pbar.update(1)
                    # Optional checkpoint logic 
                    if idx % 5000 == 0 and idx != 0:
                        EarlyStopping.save_checkpoint(self.model, self.hp.model_dir, self.hp.birdie_file_name)


                # End of epoch: compute epoch-level averages
                avg_loss = total_loss / dataset_size
                avg_mse  = total_mse_loss / dataset_size
                avg_kl   = total_kl_loss / dataset_size

            # Run validation (we'll do plain copying for a stable measure)
            val_loss, val_mse, val_kl = self._validate(val_loader)

            # Logging
            log_data = {
                "epoch": epoch + 1,
                "train_mse_loss": avg_mse,
                "train_kl_loss": avg_kl,
                "train_total_loss": avg_loss,
                "val_mse_loss": val_mse,
                "val_kl_loss": val_kl,
                "val_total_loss": val_loss
            }
            logs.append(log_data)

            print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}, MSE:{avg_mse:.4f}, KL-D:{avg_kl:.4f} || Val Loss: {val_loss:.4f}, MSE:{val_mse:.4f}, KL-D:{val_kl}")

            # Early stopping
            self.es(val_loss=val_loss, model=self.model, model_dir=self.hp.model_dir, model_file_name=self.hp.model_file_name)
            if self.es.early_stop:
                print("Early stop triggered !!")
                break
    
        # Save logs
        os.makedirs(self.hp.log_dir, exist_ok=True)
        log_path = os.path.join(self.hp.log_dir, self.hp.train_birdie_log_file)
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=4)
        print(f"Birdie training logs saved to {log_path}")



    def _validate(self, val_loader: DataLoader)->tuple:
        self.model.eval()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_kl_loss = 0.0

        dataset_size = len(val_loader.dataset)

        with torch.no_grad():
            for idx, (src, tgt) in enumerate(val_loader):
                src, tgt = src.to(self.hp.device), tgt.to(self.hp.device)
                # Only plain copying for validation
                birdie_src, birdie_tgt = copying_waveform_batch(src)

                recon_x, mu, logvar = self.model(birdie_src, birdie_tgt)
                birdie_tgt = birdie_tgt.permute(0, 2, 1)  # => [B, seq, 1]

                loss, mse, kl = self.criterion(birdie_tgt, recon_x, mu, logvar)

                batch_size = src.size(0)
                total_loss += loss.item() * batch_size
                total_mse_loss += mse.item() * batch_size
                total_kl_loss += kl.item() * batch_size


        avg_loss = total_loss / dataset_size
        avg_mse  = total_mse_loss / dataset_size
        avg_kl   = total_kl_loss / dataset_size
        return avg_loss, avg_mse, avg_kl