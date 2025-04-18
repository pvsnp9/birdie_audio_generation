from typing import Tuple
import torch

def mask_middle_waveform_batch(waveform: torch.Tensor, mask_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Birdie Objective: Infilling (Span Corruption)
    Mask a middle portion of each waveform in the batch and train the model to reconstruct it.
    
    Args:
        waveform: Tensor of shape [B, 1, T] — batch of audio waveforms
        mask_ratio: Fraction of the sequence to mask (e.g., 30%)

    Returns:
        masked_waveform: waveform with middle span zeroed out
        target_waveform: original waveform for reconstruction
    """
    B, _, T = waveform.shape
    masked_waveform = waveform.clone()
    mask_len = int(T * mask_ratio)

    for i in range(B):
        # Choose a random start index for the mask
        start = torch.randint(0, T - mask_len, (1,)).item()
        masked_waveform[i, :, start:start + mask_len] = 0.0  # silence

    return masked_waveform, waveform


def deshuffle_waveform_batch(waveform: torch.Tensor, chunk_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Birdie Objective: Deshuffling
    Shuffle fixed-size chunks of each waveform and train the model to restore original order.

    Args:
        waveform: Tensor of shape [B, 1, T]
        chunk_size: Size of each chunk to shuffle

    Returns:
        shuffled_waveform: reordered waveform chunks
        target_waveform: original waveform
    """
    B, _, T = waveform.shape
    assert T % chunk_size == 0, "Sequence length must be divisible by chunk size"

    num_chunks = T // chunk_size
    shuffled_waveform = torch.zeros_like(waveform)

    for i in range(B):
        # Break waveform into chunks
        chunks = waveform[i].unfold(1, chunk_size, chunk_size)  # [1, num_chunks, chunk_size]
        perm = torch.randperm(num_chunks)
        # Shuffle and flatten back to [1, T]
        shuffled = chunks[:, perm, :].reshape(1, -1)
        shuffled_waveform[i] = shuffled

    return shuffled_waveform, waveform


def selective_copy_waveform_batch(waveform: torch.Tensor, span_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Birdie Objective: Selective Copying
    Train model to copy a specific target span from a longer waveform.

    Args:
        waveform: Tensor of shape [B, 1, T]
        span_len: Length of the target segment to retrieve

    Returns:
        full_waveform: input with full context
        span_targets: only the target span (model is trained to generate this)
    """
    B, _, T = waveform.shape
    span_targets = torch.zeros((B, 1, span_len), dtype=waveform.dtype)

    for i in range(B):
        # Randomly choose a span inside the waveform
        start = torch.randint(0, T - span_len, (1,)).item()
        span_targets[i] = waveform[i, :, start:start + span_len]

    return waveform, span_targets  # Input is full waveform, target is only span


def copying_waveform_batch(waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Birdie Objective: Copying
    Train the model to copy the full waveform (standard autoencoder setup).

    Args:
        waveform: Tensor of shape [B, 1, T]

    Returns:
        input: waveform
        target: same waveform
    """
    return waveform, waveform
