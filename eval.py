# Standard libraries
import os
import random
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
 
# PyTorch
import torch
import torch.nn.functional as F

# Custom imports
from architectures.utils import get_model
from data.dataloader import CamusDataset


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

def export_prediction(pred, attrs, output_path, original_data_path):
    """
    Export the prediction to a MHD file, as required by the CAMUS submission platform. Since we resized the images,
    we restore the original shape and spacing by temporarily loading the original image.
    Inputs:
        pred - Prediction of the model
        attrs - Tuple containing the attributes of the data sample (patient_id, view, ED_or_ES)
        output_path - Path to the output directory
        original_data_path - Path to the original data directory
    """

    # Get the attributes
    patient_id, view, ED_or_ES = attrs
    patient_id, view, ED_or_ES = patient_id.item(), view[0], ED_or_ES[0]
    filename = f"patient{patient_id:04d}_{view}_{ED_or_ES}.mhd"

    # Load the original image
    og_image_path = os.path.join(original_data_path, f"patient{patient_id:04d}", filename)
    og_image = sitk.ReadImage(og_image_path)
    og_spacing = np.array(og_image.GetSpacing())
    og_dim = tuple(og_image.GetSize()[:2])

    # Resize the prediction to the original shape
    pred_resized = F.interpolate(pred.unsqueeze(0).type(torch.float64), size=og_dim[::-1], mode='nearest-exact')
    pred = pred_resized.cpu().numpy()[0].astype(np.uint8)
    # Convert to SimpleITK image
    pred_itk = sitk.GetImageFromArray(pred)
    assert pred_itk.GetSize() == og_image.GetSize()

    # Set spacing
    pred_itk.SetSpacing(og_spacing)

    # Export prediction
    sitk.WriteImage(pred_itk, os.path.join(output_path, filename))
    
# =============================================== MAIN ===============================================

def evaluate_test_set(model_name, model_path, test_hdf5_path = "data/camus_testing.hdf5", original_data_path = "data/testing", seed=2023):
    """
    Main function to train the model.
    Inputs:
        model_name - Name of the model to train. must be either 'unet', 'laddernet', 'enet', 'segnet', 'lednet', 'anamnet', or 'lvnet'
        checkpoint_path - Path to model directory, i.e. the one containing the tensorboard logging and the `checkpoints` folder.
        test_hdf5_path - Path to the hdf5 file containing the test data
        seed - Seed value
    """

    # Check if the inputs are valid
    assert model_name in ["unet", "laddernet", "enet", "segnet", "lednet", "anamnet", "lvnet"], \
        "Model must be either 'unet', 'laddernet', 'enet', 'segnet', 'lednet', 'anamnet', or 'lvnet'"
    assert os.path.exists(model_path), "Model path does not exist"
    assert os.path.exists(test_hdf5_path), "Test hdf5 path does not exist"
    
    # Seed everything
    seed_everything(seed)
    
    # Set the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Define output path. The predictions are saved in the model folder
    output_path = os.path.join(model_path, 'test_predictions')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Load dataset
    print("== Loading test data...")
    test_data = CamusDataset(test_hdf5_path, train_or_test="test")
    # Define DataLoaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=0)

    # Import model
    print("== Loading model checkpoint...")
    model = get_model(model_name)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoints', 'best_model.pt')))
    model.eval()

    # Run inference on the test set, and save the results
    print("== Running inference...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="== Running inference on test set...")
        for x, _, attrs in pbar:
            # send the input to the device
            x = x.to(device)
            # Run inference
            y_one_hot = model(x)
            y = torch.argmax(y_one_hot, dim=1)
            # Save the results
            export_prediction(y, attrs, output_path, original_data_path)
            
if __name__ == '__main__':
    # Parse the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="unet")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_hdf5_path', type=str, default="data/camus_testing.hdf5")
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    # Run the main function
    evaluate_test_set(args.model_name, args.model_path, args.test_hdf5_path)