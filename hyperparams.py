from dataclasses import dataclass
import torch

@dataclass
class TVAEParams:
    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hopper 
    hooper_dir: str =  f"/projects/mzampier/delta/birdie_audio_generation/"

    # Transformer architecture parameters
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 4
    seq_len: int = 1024
    window_step: int = 1

    # Training parameters
    batch_size: int = 64
    validation_size: float = 0.2
    latent_dim: int = 64
    input_size: list[int] = (1024, 1)
    input_channels: int = 1
    lr: float = 1e-5
    num_epochs: int = 50

    # Directory structure
    data_root_dir: str = f"{hooper_dir}src/data"
    data_genres_original_dir: str = f"{hooper_dir}src/data/genres_original"
    data_images_original_dir: str = f"{hooper_dir}src/data/images_original"
    spectrogram_data_dir: str = f"{hooper_dir}src/data/spectrograms"
    model_dir: str = f"{hooper_dir}src/models"
    model_file_name: str = "tvae.pth"
    birdie_file_name: str = "birdie_tvae.pth"
    output_dir: str = f"{hooper_dir}src/outputs"
    output_audio_dir: str = f"{hooper_dir}src/outputs/audio"
    audio_file_prefix: str = "generated_tvae_"
    plot_dir: str = f"{hooper_dir}src/outputs/plots"
    log_dir: str = f"{hooper_dir}src/log"
    train_log_file: str = "tvaetrainlog.json"
    train_birdie_log_file:str = "tvaebirdietrainlog.json"

    # Audio processing parameters
    sampling_rate: float = 11000 
    duration: int = 30  # seconds
    offset: float = 0.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128

    max_gen_len: int = 24000
    save_sampling_rate:int = 6000

    # run parameters 
    selected_genre: list[str] = ("rock")
