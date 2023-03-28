# Standard libraries
import os
import datetime
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
 
# PyTorch
import torch
import torch.nn as nn

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Custom imports
from eval import evaluate_test_set
from architectures.utils import get_model
from data.utils import load_train_val_with_transforms

# Path to the output of the training
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = os.path.join(SCRIPT_DIR, 'output')


# =============================================== VARIOUS FUNCTIONS ===============================================

def seed_everything(seed):
    """
    Function to set the seed for all random number generators.
    Inputs:
        seed - Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_loop(model, train_loader, optimizer, loss_module, device):
    """
    Train the model on the provided data loader.
    Inputs:
        model - PyTorch model
        data_loader - PyTorch DataLoader
        optimizer - PyTorch optimizer
        loss_module - PyTorch loss module
        device - PyTorch device
    Outputs:
        loss - Loss value of the model on the provided data
    """

    # Set the model in training mode
    model.train()
    # Initialize running average
    running_loss = 0.0

    for (i, (x, y, _)) in enumerate(train_loader):
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        # y is a tensor of shape (batch_size, 1, 512, 512). However, we need to one-hot-encode the 4 classes
        # so that the resulting tensor has shape (batch_size, 4, 512, 512).
        y_one_hot = nn.functional.one_hot(y.long(), num_classes=4).squeeze().permute(0, 3, 1, 2).float()
        loss = loss_module(pred, y_one_hot)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the running average
        running_loss += loss.item() / len(train_loader)
    
    return running_loss

def eval_loop(model, val_loader, loss_module, device):
    """
    Test the model on the provided data loader.
    Inputs:
        model - PyTorch model
        data_loader - PyTorch DataLoader
        loss_module - PyTorch loss module
        device - PyTorch device
    Outputs:
        loss - Loss value of the model on the provided data
    """

    # Initialize running average
    running_loss = 0.0

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y, _) in val_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            y_one_hot = nn.functional.one_hot(y.long(), num_classes=4).squeeze().permute(0, 3, 1, 2).float()
            # make the predictions and calculate the validation loss
            pred = model(x)
            loss = loss_module(pred, y_one_hot)
            # update the running average
            running_loss += loss.item() / len(val_loader)

    return running_loss

# =============================================== MAIN FUNCTION ===============================================

def main(args):
    """
    Inputs:
        args - Namespace object from the argument parser
    """

    # Check if the model is valid
    assert args.model in ["unet", "laddernet", "enet", "segnet", "lednet", "anamnet", "lvnet"], \
        "Model must be either 'unet', 'laddernet', 'enet', 'segnet', 'lednet', 'anamnet', or 'lvnet'"
    
    # Set the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Set the seed 
    seed_everything(args.seed)

     # Prepare logging
    if args.save: 
        model_dir = os.path.join(DEST_DIR, args.model)
        experiment_dir = os.path.join(model_dir, f"{datetime.datetime.now().strftime('date_%d_%m_%Y__time_%H_%M_%S')}")                      # Date and time
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(
            experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(experiment_dir)

    # Define the transforms
    if args.transform:
        train_transform = A.Compose(
            [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ]
        )
    else:
        train_transform = None

    transformations = {
        'train': train_transform,
        'val': None,
    }

    print("== Loading data...")
    # Load dataset
    train_set, val_set = load_train_val_with_transforms(file_path=args.data_path, transforms=transformations, val_set_size=0.2)
    # Define DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, persistent_workers=True if args.num_workers > 0 else False, num_workers=args.num_workers, pin_memory=True)

    print("== Initializing model and optimizer...")
    # Initialize the model
    model = get_model(args.model)
    model = model.to(device)
    # Create optimizer and loss module
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    loss_module = nn.CrossEntropyLoss()
    
    # calculate steps per epoch for training and test set
    # initialize a dictionary to store training history
    logging = {"train_loss": [], "test_loss": []}

    # Keep track of best val loss to save best model
    best_val_loss = np.inf
    
    print("== Starting training loop...")
    pbar = tqdm(range(args.num_epochs), desc="Training model...")
    for e in pbar:

        ##############
        #  TRAINING  #
        ##############
        train_loss = train_loop(model, train_loader, optimizer, loss_module, device)
        logging["train_loss"].append(train_loss)

        ##############
        # VALIDATION #
        ##############
        val_loss = eval_loop(model, val_loader, loss_module, device)
        logging["test_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model.pt"))

        # Tensorboard logging
        if args.save:
            writer.add_scalar('train_loss', train_loss, e)
            writer.add_scalar('val_loss', val_loss, e)

        # Update pbar description
        pbar.set_description(f"Training model... | Epoch: {e:03d} | Train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f})")
    
    if args.test:
        print("== Testing model...")
        evaluate_test_set(args.model_name, experiment_dir, test_hdf5_path = "data/camus_testing.hdf5", original_data_path = "data/testing", seed=args.seed)
        
# =============================================== ARG PARSING ===============================================

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Various training parameters
    parser.add_argument('--model', type=str, required=True,
                        help="Model to use. Model must be either 'unet', 'laddernet', 'enet', 'segnet', 'lednet', 'anamnet', or 'lvnet'")
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Seed for random number generators')
    
    # Whether to evaluate the model on the test set at the end of training
    parser.add_argument('--test', action='store_true',
                        help='Whether to evaluate the model on the test set at the end of training.')
    parser.add_argument('--no_test', dest='test', action='store_false', 
                        help='Whether to avoid evaluating the model on the test set at the end of training.')
    parser.set_defaults(test=True)
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=r'data/camus_training.hdf5',
                        help='Path to the training Camus dataset in HDF5 format')
    parser.add_argument('--num_workers', default=24, type=int,
                        help='Number of workers to use for data loading')
    
    # Data augmentation parameters
    parser.add_argument('--transform', action='store_true',
                        help='Whether to use data transformations on the train set.')
    parser.add_argument('--no_transform', dest='transform', action='store_false',
                        help='Whether to avoid using data transformations on the train set.')
    parser.set_defaults(transform=True)

    # Whether to save the models
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the outputs of the training.')
    parser.add_argument('--no_save', dest='save', action='store_false',
                        help='Whether to avoid saving the outputs of the training.')
    parser.set_defaults(save=True)

    args = parser.parse_args()

    # Run the training
    main(args)