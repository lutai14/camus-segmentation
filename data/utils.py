import torch
from .dataloader import CamusDataset

def load_train_val_with_transforms(file_path, transforms, val_set_size, verbose = True):
    """
    Load and split dataset into train and validation set using the specified val set size. Use the specified transforms for train and validation set.
    Useful e.g. if you only want to apply transform to the training set.
    Inspired by https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/8
    args:
        file_path: path to the dataset in HDF5 format
        transforms: dictionary containing the transforms for train and validation set
        val_set_size: if `float`, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
                      If `int`, represents the absolute number of test samples. If None, the value is set to 0.25.
        verbose: print additional information
    returns:
        train_dataset: training set
        val_dataset: validation set
    """

    # Load dataset
    train_dataset = CamusDataset(file_path=file_path, transform=transforms['train'])
    val_dataset =   CamusDataset(file_path=file_path, transform=transforms['val'])
    if verbose: print(f"Loaded {len(train_dataset)} samples from {train_dataset.no_unique_patients} patients" )
    
    # Determine size of validation set
    if isinstance(val_set_size, float):
        val_set_size = int(len(train_dataset) * val_set_size)
    elif isinstance(val_set_size, int):
        pass

    # Split dataset into train and validation set
    indices = torch.randperm(len(train_dataset), generator=torch.Generator()).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_set_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_set_size:])

    if verbose: print(f"Created a training set of {len(train_dataset)} image pairs and a validation set of {len(val_dataset)}." )
    
    return train_dataset, val_dataset