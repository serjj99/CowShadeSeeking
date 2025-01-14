from torch.utils.data import DataLoader, TensorDataset 
from torch import tensor, float32


def torch_DataLoader(X, Y, batch_size,
                     device='cpu', drop_last=False, shuffle=True):
    """
    Create tensor data loaders for training and validation in PyTorch.

    Args:
        X: Features for training
        Y: Targets for training
        batch_size: Batch size
        device: Device to use for tensors
        drop_last (boolean): Whether to drop the last incomplete batch
        shuffle (boolean): Whether to shuffle the DataLoader each epoch
    """
    
    train_dataset = TensorDataset(
        tensor(X, device=device, dtype = float32),
        tensor(Y, device=device, dtype = float32)
    )

    data_loader = DataLoader(train_dataset,
                             batch_size = batch_size,
                             drop_last  = drop_last,
                             shuffle    = shuffle)

    return data_loader