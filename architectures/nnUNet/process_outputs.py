import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

def convert_for_submission(predictions_path, original_test_path):
    """
    Convert the outputs of nnUNet to the format required for submission (MHD files). The converted files
    will be saved in `nnUNet/submissions`.
    args:
        predictions_path: Path to the folder containing the predictions
        original_test_path: Path to the original test set
    """

    # Create a directory in the `submission` folder, named as the task
    output_path = "submissions/"
    submission_path = os.path.join(output_path, os.path.basename(predictions_path))
    os.makedirs(submission_path, exist_ok=True)

    pred_files = [file for file in os.listdir(predictions_path) if file.endswith(".nii.gz")]

    for prediction in pred_files:
        # Retrieve original test file to resize prediction and set the correct spacing
        patient = prediction.split("_")[0]
        test_filepath = os.path.join(original_test_path, patient, prediction.replace(".nii.gz", ".mhd"))
        original = sitk.ReadImage(test_filepath)

        og_spacing = np.array(original.GetSpacing())
        og_dim = tuple(original.GetSize()[:2])

        # Resize to original shape
        pred_resized = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(predictions_path, prediction)))).unsqueeze(0)
        pred_resized = F.interpolate(pred_resized, size=og_dim[::-1], mode='nearest-exact')
        pred = pred_resized.numpy()[:,0]

        # Convert to SimpleITK image
        pred = sitk.GetImageFromArray(pred)

        assert pred.GetSize() == original.GetSize()

        # Set spacing
        pred.SetSpacing(og_spacing)

        # Export prediction
        filename = prediction.replace(".nii.gz", ".mhd")
        sitk.WriteImage(pred, os.path.join(submission_path, filename))


if __name__ == "__main__":

    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--predictions_path", type=str, default="./output/Task570_CAMUS", help="Path to the folder containing the predictions")
    args.add_argument("--original_test_path", type=str, default="../../data/testing", help="Path to the original test set")
    args = args.parse_args()

    # Convert .nii.gz to .mhd for submission platform
    convert_for_submission(args.predictions_path, args.original_test_path)
