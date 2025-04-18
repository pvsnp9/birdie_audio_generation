import torch 
import torch.nn as nn 
from typing import Tuple, Optional
from latent_enc_dec import *
from hyperparams import TVAEParams
from utils import get_sinusoidal_positional_encoding
import os


###############################################################################
# Transformer CrossAttentionVAE
###############################################################################
class TransformerCrossAttentionVAE(nn.Module):
    def __init__(self, hp: TVAEParams):
        super().__init__()
        self.hp = hp
        self.checkpoint_loaded = False

        self.encoder = LatentEncoder(hp)
        self.latent_head = LatentHead(hp)
        self.latent_conditioner = LatentConditioner(hp)
        self.decoder = LatentDecoder(hp)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = mu + sigma * eps
        """
        eps = torch.randn_like(mu)
        sigma = (0.5 * logvar).exp()
        return mu + sigma * eps

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        src: [B, 1, seq_len] input waveforms
        tgt: [B, 1, seq_len] target waveforms for reconstruction
        Returns: (reconstruction, mu, logvar)
        """
        # Encode full seq => [seq_len, B, d_model]
        encoder_out = self.encoder(src)

        # Get latent from entire sequence
        mu, logvar = self.latent_head(encoder_out)
        z = self.reparameterize(mu, logvar)

        # Condition the memory with z
        memory = self.latent_conditioner(encoder_out, z)

        # Decode 
        reconstruction = self.decoder(memory, tgt)
        return reconstruction, mu, logvar
    
    def load_checkpoint(self, train_type= "birdie"):
        try:
            model_selection = self.hp.birdie_file_name if train_type == "birdie" else self.hp.model_file_name
            print(f"Loading {model_selection} from {self.hp.model_dir}")
            if not self.checkpoint_loaded:
                checkpoint = torch.load(os.path.join(self.hp.model_dir, model_selection),weights_only=True, map_location=self.hp.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.load_state_dict(checkpoint["model_state_dict"])
                else:
                    # If the file is just a state_dict without wrapping
                    self.load_state_dict(checkpoint)
                
                self.eval()
                self.to(self.hp.device)
                self.checkpoint_loaded = True
                print(f"Checkpoint loaded successfully from  {self.hp.model_dir}/{model_selection}")

            else: print(f"Checkpoint already loaded from {self.hp.model_dir}/{model_selection}")

        except Exception as e:
            self.checkpoint_loaded = False
            print(f"Error loading checkpoint from {self.hp.model_dir}/{model_selection}")
    
    def generate_from_latent(self, z: torch.Tensor, max_len: int = None) -> torch.Tensor:
        if max_len is None:
            max_len = self.hp.seq_len

        # We skip the encoder; we just broadcast z across the memory
        # For cross-attention, we need an artificial “dummy” encoder_out or a single vector repeated
        dummy_enc_out = torch.zeros(self.hp.seq_len, z.size(0), self.hp.d_model, device=z.device)
        memory = self.latent_conditioner(dummy_enc_out, z)

        # Create dummy target for the decoder
        B = z.size(0)
        dummy_tgt = torch.zeros(B, 1, max_len, device=z.device)
        generated = self.decoder(memory, dummy_tgt)  # => [B, max_len, 1]
        return generated


    def generate_conditional(self, src: torch.Tensor, max_len: int = None) -> torch.Tensor:
        """
        Generate a waveform using the model, given an input 'src'.
        This runs the encoder on 'src', samples a latent z, then decodes
        a dummy (all-zero) target of length 'max_len'.

        Args:
            src: [B, 1, seq_len] input waveforms used for conditioning
            max_len: optional length for the generated output.
                     defaults to self.hp.seq_len if None.

        Returns:
            generated: [B, max_len, 1] a newly generated waveform
        """
        if max_len is None:
            max_len = self.hp.seq_len

        # Encode the source input
        encoder_out = self.encoder(src)  # => [seq_len, B, d_model]

        # Sample a latent from the VAE
        mu, logvar = self.latent_head(encoder_out)
        z = self.reparameterize(mu, logvar)

        # Condition the entire memory with z
        memory = self.latent_conditioner(encoder_out, z)

        # Create a dummy target for the decoder
        #    We'll fill it with zeros and let the decoder produce 'max_len' steps
        B = src.size(0)
        dummy_tgt = torch.zeros(B, 1, max_len, device=src.device)  # shape [B, 1, max_len]

        # Decode
        generated = self.decoder(memory, dummy_tgt)  # => [B, max_len, 1]
        return generated


# if __name__ == "__main__":
#     # Instantiate hyperparams
#     hp = TVAEParams()

#     # Create model
#     model = TransformerCrossAttentionVAE(hp).to(hp.device)

#     # Dummy batch [B=4, C=1, seq_len=1024]
#     dummy_src = torch.randn(4, 1, hp.seq_len).to(hp.device)
#     dummy_tgt = torch.randn(4, 1, hp.seq_len).to(hp.device)

#     # Forward pass
#     recon, mu, logvar = model(dummy_src, dummy_tgt)
#     print("Recon shape:", recon.shape)     # Expect [4, 1024, 1]
#     print("mu, logvar shapes:", mu.shape, logvar.shape)