import matplotlib.pyplot as plt
import os
from hyperparams import *
import seaborn as sns
from scipy.stats import norm
import numpy as np
import torchaudio

def visualize_latent_distribution(z, plot_dims=64, hp=None, save_path=None):
    # Number of dimensions in the latent vector
    latent_dim = z.shape[1]
    
    # We'll plot up to `plot_dims` or the actual latent_dim, whichever is smaller
    dims_to_plot = min(plot_dims, latent_dim)
    
    # Prepare a range and a standard normal PDF for overlay
    x = np.linspace(-4, 4, 1000)
    gaussian = norm.pdf(x)
    
    # Style settings
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", n_colors=1)
    
    # Create the figure. 8x8 subplots if plot_dims=64.
    plt.figure(figsize=(20, 15))
    
    for idx in range(dims_to_plot):
        plt.subplot(8, 8, idx + 1)
        
        # Plot empirical distribution using KDE
        sns.kdeplot(z[:, idx],
                    fill=True,
                    alpha=0.4,
                    linewidth=2,
                    color=palette[0],
                    label="Empirical")
        
        # Overlay standard normal
        plt.plot(x, gaussian, '--', color='#2e3440', linewidth=1.5, label="N(0,1)")
        
        # X-axis limit for better visual comparison
        plt.xlim(-4, 4)
        
        # A bit of grid
        plt.grid(True, alpha=0.3)
        
        # Title each subplot as Dim i
        plt.title(f"Dim {idx}", fontweight='semibold')
        
        # Hide y-label text for a cleaner look (optional)
        plt.ylabel("")
        
        # Show legend only on the first subplot
        if idx == 0:
            plt.legend(fontsize=9, frameon=True, shadow=True)
    
    # Add an overall title
    plt.suptitle("Latent Space Analysis of TVAE Birdie Model", y=1.02, fontsize=16, fontweight='bold')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    if save_path:
        # Ensure the output directory exists.
        os.makedirs(hp.plot_dir, exist_ok=True)
        fig_path = os.path.join(hp.plot_dir, save_path)
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
    
    # plt.show()


def find_files_from_dir(audio_dir:str)->tuple:
    birdie_file = None
    tvae_file = None

    # List all .wav files in directory
    for fname in os.listdir(audio_dir):
        if not fname.lower().endswith(".wav"):
            continue

        fpath = os.path.join(audio_dir, fname)
        # Check filename
        if "birdie" in fname.lower():
            birdie_file = fpath
        elif "tvae" in fname.lower():
            tvae_file = fpath

    return birdie_file, tvae_file

def read_audio_mono(audio_path:str)->tuple:
    waveform, sr = torchaudio.load(audio_path)
    # waveform shape: [channels, num_samples]
    if waveform.shape[0] > 1:
        # average across channels => mono
        waveform = waveform.mean(dim=0, keepdim=True)
    # shape => [1, num_samples]
    wave_np = waveform[0].cpu().numpy()  # [num_samples]
    return wave_np, sr


def read_and_plot_audio(audio_dir: str, output_path: str, figsize=(10,4)) -> None:
    birdie_file, tvae_file = find_files_from_dir(audio_dir)
    if not birdie_file or not tvae_file:
        print("Could not find both 'birdie' and 'tvae' files in", audio_dir)
        return

    birdie_wave, sr_b = read_audio_mono(birdie_file)
    tvae_wave, sr_t = read_audio_mono(tvae_file)


    time_b = np.linspace(0, len(birdie_wave)/sr_b, num=len(birdie_wave))
    time_t = np.linspace(0, len(tvae_wave)/sr_t, num=len(tvae_wave))

    # 4) Plot
    plt.figure(figsize=figsize)

    # Birdie waveform
    plt.subplot(2,1,1)
    plt.plot(time_b, birdie_wave, color="blue")
    plt.title(f"TVAE BIRDIE: {os.path.basename(birdie_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # TVAE waveform
    plt.subplot(2,1,2)
    plt.plot(time_t, tvae_wave, color="green")
    plt.title(f"TVAE Regular: {os.path.basename(tvae_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")
