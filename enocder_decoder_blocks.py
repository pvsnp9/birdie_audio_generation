import torch
import torch.nn as nn
from hyperparams import TVAEParams
from typing import Tuple
from utils import get_sinusoidal_positional_encoding

class Encoder(nn.Module):
    def __init__(self, hp:TVAEParams):
        super().__init__()
        self.hp = hp
        # project input channel to d_model 
        self.input_projection = nn.Linear(hp.input_channels, hp.d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=hp.d_model, nhead=hp.n_heads, dim_feedforward=4*hp.d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=hp.num_layers)
        # projection to latent dim
        self.mu = nn.Linear(hp.d_model, hp.latent_dim)
        self.logvar = nn.Linear(hp.d_model, hp.latent_dim)

    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        # X: [B, SEQ_LEN, Features] but we have [B, Feature, SEQ]. lets chenge that. 
        x = x.permute(0, 2, 1) # [B, Sq, Ft] [B, seq_len, 1]
        # project input to d_model
        x = self.input_projection(x) # [B, Sq, d_model]

        batch_size = x.size(0)
        special_tokens = torch.zeros(batch_size, 1, x.size(2), device=x.device)
        # add token to data 
        x = torch.cat([special_tokens, x], dim=1) # [B, seq_len + 1, d_model]
        pos_enc = get_sinusoidal_positional_encoding(x.size(1), self.hp.d_model, device=x.device) # [B, seq_len + 1, d_model]
        # add positional encodings 
        x = x + pos_enc.unsqueeze(0) #[1, seq_len + 1, d_model]
        # since transformer expects [Sq, B, d_model]
        x = x.transpose(0,1)
        encoder_out = self.transformer_encoder(x)
        # Take output of special token (position 0)
        bottleneck = encoder_out[0] # [B, d_model]
        mu = self.mu(bottleneck)
        logvar = self.logvar(bottleneck)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, hp:TVAEParams):
        super().__init__()
        self.hp = hp
        self.latent_transformation = nn.Linear(hp.latent_dim, hp.d_model) 
        self.target_projection = nn.Linear(hp.input_channels, hp.d_model) 
        dec_layer = nn.TransformerDecoderLayer(d_model=hp.d_model, nhead=hp.n_heads, dim_feedforward=4*hp.d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=hp.num_layers)
        # output proejction to single value
        self.out_projection = nn.Linear(hp.d_model, 1)
    
    def forward(self, z:torch.Tensor, tgt:torch.Tensor)->torch.Tensor:
        # Z [B, latent_dim]
        # project Latent_dim to model_dim
        latent = self.latent_transformation(z) #[B, latent_dim] -> [B, d_model]
        memory = latent.unsqueeze(0)  # Shape: (1, batch, d_model)
        # prep target sequence to be filled with pad (zeros)
        # reshape the data [B, C, Seq] -> [B, Seq, C]
        tgt = tgt.permute(0, 2, 1)
        tgt = self.target_projection(tgt) # [B, Seq, d_model]
        pos_enc = get_sinusoidal_positional_encoding(tgt.size(1), self.hp.d_model, device=tgt.device) # [B, seq_len + 1, d_model]
        target = tgt + pos_enc.unsqueeze(0) 
        target = target.transpose(0, 1) # [seq_len, B, d_model]
        # create a causal mask (AR property)
        tatrget_mask = nn.Transformer.generate_square_subsequent_mask(self.hp.input_size[0]).to(z.device)
        # decode the sequence using cross-attention with memory
        decoder_out = self.transformer_decoder(tgt=target, memory=memory, tgt_mask=tatrget_mask) # [Seq, B, d_model]
        decoder_out = decoder_out.permute(1, 0, 2) # [B, Seq, d_model]
        output = self.out_projection(decoder_out) # [B, Seq, 1]
        return output



