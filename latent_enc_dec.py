import torch
import torch.nn as nn
from hyperparams import TVAEParams
from typing import Tuple
from utils import get_sinusoidal_positional_encoding

###############################################################################
# LatentEncoder: processes entire audio -> returns [seq_len, B, d_model]
###############################################################################
class LatentEncoder(nn.Module):
    def __init__(self, hp: TVAEParams):
        super().__init__()
        self.hp = hp
        # Project input (1 channel) to d_model
        self.input_projection = nn.Linear(hp.input_channels, hp.d_model)
        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model,
            nhead=hp.n_heads,
            dim_feedforward=4 * hp.d_model,
            batch_first=False  # PyTorch default for Transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=hp.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, seq_len] or [B, input_channels, seq_len]
        We want output: [seq_len, B, d_model]
        """
        # Permute to [B, seq_len, channels]
        x = x.permute(0, 2, 1)  # => [B, seq_len, 1]
        # Project to [B, seq_len, d_model]
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)  # e.g. 1024
        pos_enc = get_sinusoidal_positional_encoding(seq_len, self.hp.d_model, x.device)  # [seq_len, d_model]
        # x => [B, seq_len, d_model], pos_enc => [seq_len, d_model]
        # We'll broadcast-add after transposing x => [seq_len, B, d_model]
        x = x.transpose(0,1)  # => [seq_len, B, d_model]
        x = x + pos_enc.unsqueeze(1)  # broadcasting [seq_len, 1, d_model]

        # Pass through Transformer Encoder
        encoder_out = self.transformer_encoder(x)  # => [seq_len, B, d_model]
        return encoder_out



###############################################################################
# LatentHead: aggregates encoder_out -> produce (mu, logvar)
###############################################################################
class LatentHead(nn.Module):
    def __init__(self, hp: TVAEParams):
        super().__init__()
        self.hp = hp
        self.mu = nn.Linear(hp.d_model, hp.latent_dim)
        self.logvar = nn.Linear(hp.d_model, hp.latent_dim)

    def forward(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        encoder_out: [seq_len, B, d_model]
        We'll mean-pool across seq_len dimension.
        """
        # average across seq_len => [B, d_model]
        pooled = encoder_out.mean(dim=0)
        mu = self.mu(pooled)         # => [B, latent_dim]
        logvar = self.logvar(pooled) # => [B, latent_dim]
        return mu, logvar
    

###############################################################################
# LatentConditioner: broadcast latent z to entire encoder_out
###############################################################################
class LatentConditioner(nn.Module):
    def __init__(self, hp: TVAEParams):
        super().__init__()
        self.hp = hp
        # Project z => d_model
        self.z_to_dm = nn.Linear(hp.latent_dim, hp.d_model)

    def forward(self, encoder_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        encoder_out: [seq_len, B, d_model]
        z: [B, latent_dim]
        We'll broadcast z across seq_len dimension by adding or gating.
        """
        # shape => [B, d_model]
        z_proj = self.z_to_dm(z)
        # expand z_proj to [1, B, d_model], then add to each time-step
        z_proj = z_proj.unsqueeze(0)  # => [1, B, d_model]
        memory = encoder_out + z_proj # => [seq_len, B, d_model]
        return memory
    


###############################################################################
# LatentDecoder: cross-attends to the entire memory
###############################################################################
class LatentDecoder(nn.Module):
    def __init__(self, hp: TVAEParams):
        super().__init__()
        self.hp = hp
        # Project input (1 channel) to d_model
        self.target_projection = nn.Linear(hp.input_channels, hp.d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hp.d_model,
            nhead=hp.n_heads,
            dim_feedforward=4 * hp.d_model,
            batch_first=False  # must stay False for standard PyTorch
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=hp.num_layers)
        # Finally project to 1 channel
        self.output_projection = nn.Linear(hp.d_model, hp.input_channels)

    def forward(self, memory: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        memory: [seq_len, B, d_model] from the encoder
        tgt: [B, 1, seq_len] -> we project, add PE, then decode
        Returns: [B, seq_len, 1]
        """
        # Reshape tgt => [B, seq_len, 1]
        tgt = tgt.permute(0, 2, 1)  # => [B, seq_len, 1]
        tgt = self.target_projection(tgt)  # => [B, seq_len, d_model]

        # Add positional encoding
        seq_len = tgt.size(1)
        pos_enc = get_sinusoidal_positional_encoding(seq_len, self.hp.d_model, tgt.device)  # [seq_len, d_model]
        tgt = tgt.transpose(0,1)  # => [seq_len, B, d_model]
        tgt = tgt + pos_enc.unsqueeze(1)  # => [seq_len, B, d_model]

        # Causal mask for autoregression, if desired
        # Usually we do next-sample or next-chunk. If you want a full teacher-forcing approach:
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # Cross-attend
        decoder_out = self.transformer_decoder(tgt, memory, tgt_mask=mask)
        # => [seq_len, B, d_model]

        # Reshape to [B, seq_len, d_model]
        decoder_out = decoder_out.transpose(0,1)
        # Project to [B, seq_len, 1]
        out = self.output_projection(decoder_out)
        return out