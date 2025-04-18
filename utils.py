import librosa
import os
import torchaudio
from datetime import datetime
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from hyperparams import  TVAEParams
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
# import soundfile as sf
import math
import seaborn as sns
from scipy.stats import norm

def get_audio_metadata(file_path:str)->tuple:
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        channels = y.shape[0] if y.ndim > 1 else 1
        duration = librosa.get_duration(y=y, sr=sr)
        return sr, channels, duration
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def plot_generated_audio(audio: torch.Tensor, sample_rate: int = 3000) -> None:
    """
    Plots generated audio samples.

    Args:
        audio (torch.Tensor): The generated audio tensor. It can have one of these shapes:
                              - (length,) for a single-channel, single-sample signal,
                              - (channels, length) for a multi-channel single-sample signal (plots first channel),
                              - (num_samples, channels, length) for a batch of samples (plots first sample, first channel).
        sample_rate (int): Sample rate of the audio used to create the time axis.
    """
    # Handle different tensor dimensions.
    if audio.ndim == 3:
        # Audio has shape (num_samples, channels, length).
        num_samples, channels, length = audio.shape
        # For demonstration, we plot the first sample's first channel.
        waveform = audio[0, 0].cpu().numpy()
        title = "Generated Audio (Sample 0, Channel 0)"
    elif audio.ndim == 2:
        # Audio has shape (channels, length) for a single sample.
        channels, length = audio.shape
        waveform = audio[0].cpu().numpy()  # Plot the first channel.
        title = "Generated Audio (Channel 0)"
    elif audio.ndim == 1:
        # Audio has shape (length,)
        length = audio.shape[0]
        waveform = audio.cpu().numpy()
        title = "Generated Audio"
    else:
        raise ValueError("Audio tensor has an unsupported number of dimensions.")
    
    # Create a time axis.
    time_axis = np.linspace(0, length / sample_rate, num=length)
    
    # Plot the waveform.
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def vae_loss(x: torch.Tensor,  x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta:float = 1.0) -> Tuple[torch.Tensor, ...]:
    # Reconstruction loss: mean squared error
    mse_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence loss: sum over latent dims, then average over batch.
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = mse_loss + beta * kl_loss
    return total_loss, mse_loss, kl_loss


def get_sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device = None) -> torch.Tensor:
    """
    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimensionality of the model.
        device (torch.device, optional): Device to place the tensor.
    
    Returns:
        Tensor of shape [seq_len, d_model] with positional encodings.
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if device is not None:
        pe = pe.to(device)
    return pe

def plot_and_save_latent_distribution(z:np.ndarray, hp:TVAEParams):
    ncols = 8
    
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.8)

    n_samples, latent_dim = z.shape
    nrows = math.ceil(latent_dim / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=False, sharey=False)
    axes = axes.flatten()  # so we can index axes[i]

    for i in range(latent_dim):
        ax = axes[i]
        data = z[:, i]  # [N] one latent dimension across all samples
        
        # Plot the empirical distribution (kde)
        sns.kdeplot(data, fill=True, color="salmon", alpha=0.6, label="Empirical", ax=ax)

        # Overlay a standard Normal(0,1)
        x_vals = np.linspace(-4, 4, 200)
        std_pdf = norm.pdf(x_vals, 0, 1)  # mean=0, std=1
        ax.plot(x_vals, std_pdf, "k--", linewidth=2, label="N(0,1)")

        ax.set_title(f"Dim {i}")
        ax.legend(loc="upper left")
    
    # Hide unused subplots if latent_dim doesn't fill the grid
    for j in range(latent_dim, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(hp.plot_dir), exist_ok=True)
    
    plt.savefig(f'{hp.plot_dir}/latent_plot.png', dpi=150)
    plt.close(fig)
    print(f"Saved latent distribution figure to {hp.plot_dir}")


# def save_generated_audio(model: any, hp:TVAEParams, n_batches: int = 1, channel: int = 1) -> None:
#     """
#     Args:
#         model (TVAE): The trained TVAE model.
#         n_batches (int): Number of audio batches (files) to generate.
#         channels (int): Number of audio samples per batch or channels in one audio
#     """
#     # Ensure output directory exists.
#     os.makedirs(hp.output_audio_dir, exist_ok=True)
#     model.eval()  # Set model to evaluation mode
#     max_len = 6000
#     save_sampling_rate = 3000
#     with torch.no_grad():
#         for i in range(n_batches):
#             try:
#                 # The output shape [batch_size, max_len, 1]
#                 generated = model.generate_music(hp=hp, batch_size=channel, max_len=max_len)
                
#                 # Convert tensor to NumPy array and squeeze the last dimension.
#                 # [channel, max_len, 1] -> [channel, max_len]
#                 generated_np = generated.squeeze(-1).cpu().numpy()
                
#                 # For saving as a WAV file, we want shape [max_len, channels]
#                 # Transpose
#                 generated_np = generated_np.T  # [max_len, channel]
                
#                 now = datetime.now()

#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 # Define output file path.
#                 output_path = os.path.join(hp.output_audio_dir, f"tave_generated_audio_batch_{timestamp}.wav")
                
#                 # Write the WAV file.
#                 sf.write(output_path, generated_np, samplerate=save_sampling_rate)
#                 print(f"Saved generated audio batch {i+1} to {output_path}")
#                 return generated
#             except Exception as e:
#                 print(f"Error during generation or saving batch {i+1}: {e}")




# if __name__ == "__main__":
    # Initialize hyperparameters and model.
    # hp = TVAEParams()
#     device = hp.device
#     model = TVAE(hp).to(device)
    
#     # Optionally load a checkpoint here:
#     # model.load_checkpoint("path/to/checkpoint.pth")
    
#     # Generate and save 3 batches (each with 5 audio samples of length 8000)
#     save_generated_audio(model,hp=hp)