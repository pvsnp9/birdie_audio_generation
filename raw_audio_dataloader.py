from typing import Tuple
from torch.utils.data import DataLoader, random_split
from hyperparams import TVAEParams
from raw_dataset import TVAESeqSeqDataset

"""TAVE data loader"""
def get_tvae_dataloaders(hp: TVAEParams, num_workers=1) -> Tuple[DataLoader, DataLoader]:
    dataset = TVAESeqSeqDataset(hp=hp)
    total_samples: int = len(dataset)
    val_size: int = int(total_samples * hp.validation_size)
    train_size: int = total_samples - val_size

    # Split dataset randomly
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    try:
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=num_workers)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=num_workers)
    except Exception as e:
        raise RuntimeError(f"Error creating DataLoaders: {e}")
    
    return train_loader, val_loader


# if __name__ == '__main__':
# #       TVAE TEST
#         hp =TVAEParams()
#         hp.batch_size = 4
        
#         try:
#             train_loader, val_loader = get_tvae_dataloaders(hp=hp)
#         except Exception as e:
#             print(e)
        
#         try:
#             for batch_waveforms, batch_labels in train_loader:
#                 print("Batch waveforms shape:", batch_waveforms.shape)  # Expected: (batch, channels, segment_length)
#                 print("Batch labels:", batch_labels.shape)
#                 break  # Remove break to iterate through full loader
#         except Exception as e:
#             print(f"Error during iteration: {e}")