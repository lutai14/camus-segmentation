import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd


class CamusDataset(Dataset):
    """
    Dataloader for the unsupervised training on the Camus dataset based on the hdf5 file created by `camus_hdf5_conversion.py`.
    It creates a csv file for easy lookup of the data.
    """
    def __init__(self, file_path, transform=None):
        """
        Inputs:
            file_path - Path to the hdf5 file containing the Camus dataset
            transform - Albumentations transformation to apply to the images
        """

        assert os.path.exists(file_path), f"File {file_path} does not exist. Please run the `data_preparation.py` script in the /camus directory to create it."

        super().__init__()
        self.file = None
        self.file_path = file_path
        self.transform = transform
        self.df = pd.DataFrame(columns=['patient', 'id', 'view', 'ED/ES'])

        df_frames_dir = os.path.join(os.path.dirname(self.file_path), 'lookup_tables')
        df_frames_path = os.path.join(df_frames_dir, f"df_frames.csv")

        create = True

        # If the dataframe already exists (with the correct amount of leave_out_patients) we load it
        if os.path.exists(df_frames_path):	
            self.df = pd.read_csv(df_frames_path)
            create = False

        # Otherwise we create it.
        # Because an opened HDF5 file isn’t pickleable and to send Dataset to workers’ processes it needs to be serialised with pickle, you can’t 
        # open the HDF5 file in __init__. Open it in __getitem__ and store as the singleton!. Do not open it each time as it introduces huge overhead.
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643
        # However, we temporarily open the file to generate the dataframe with all frames of all patients
        with h5py.File(self.file_path, 'r') as file:
            if create:
                if not os.path.exists(df_frames_dir):
                    os.makedirs(df_frames_dir)

                print(f"CSV file could not be found at path `{df_frames_path}`. Generating it now. This may take a while...")

                for patient in file.keys():
                        patient_id = int(patient[-4:])
                        for view in file[patient].keys():
                            ED_frame = file[f'{patient}/{view}'].attrs['ED']                            
                            ES_frame = file[f'{patient}/{view}'].attrs['ES']

                            patient_df = pd.DataFrame({
                                'patient': [patient]*2, 
                                'id': [patient_id]*2, 
                                'view': [view]*2, 
                                'ED/ES': ['ED', 'ES'], 
                                'frame_no': [ED_frame, ES_frame]},
                                index=[0,1])
                            self.df = pd.concat([self.df, patient_df], ignore_index=True)

                # Save csv           
                self.df.to_csv(df_frames_path, index = False)
                    
        self.no_unique_patients = len(set(self.df['patient']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError # Preventing out of bounds, as seen here https://stackoverflow.com/questions/54640906/torch-dataset-looping-too-far
       
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        patient, patient_id, view, ED_or_ES, frame_no = list(self.df.loc[idx])

        # Load image with shape (W,H)
        image = self.file[f'{patient}/{view}/data'][int(frame_no-1)]
        mask = self.file[f'{patient}/{view}/masks'][0 if ED_or_ES == 'ED' else 1]

        if self.transform is not None:
            # Transformations are implemented using Albumentations
            image = self.transform(image=image)['image']

        # We return the image as a 3D tensor with shape (1, H, W) to be compatible with the rest of the code
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), (patient_id, view, ED_or_ES)

    def select_transforms(self, dataset):
        # Define whether we are using the training or validation set, to correctly select the desired transfromations.
        self.dataset = dataset