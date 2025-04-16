import torch
import pickle
import pandas as pd
import numpy as np
import os
import nibabel as nib

from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset

# ADNI dataset class
class ADNIDataset4D(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.csv_path = config['adni_csv']
        self.dataset_path = config['adni_train_path'] if mode == 'train' else config['adni_val_path']
        
        # self.generate_data(config['adni_train_path'], config['adni_val_path'])
        # self.generate_folds('./src/data/')
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        # Data filtering
        self.data = [sample for sample in self.data if sample[3] < 68 or sample[3] > 80]
        # self.data = [sample for sample in self.data if sample[1] in ['AD', 'CN']]
        # self.data = self.data[:int(len(self.data) * 0.1)]

        print(f"Dataset initialized: {len(self.data)} {mode} samples")

    def generate_data(self, train_path, val_path):
        # Load CSV data
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Group', 'Sex', 'Age', 'Path_sMRI_brain', 'Path_fMRI_brain'])
        print(f"Total rows in CSV: {len(df)}")              # 690
        
        # Get unique subjects and their counts
        unique_subjects = df['Subject'].unique()
        n_subjects = len(unique_subjects)
        print(f"Total unique subjects: {n_subjects}")       # 206
        
        # Randomly shuffle and split subjects
        shuffled_subjects = np.random.permutation(unique_subjects)
        train_size = int(0.8 * n_subjects)  
        train_subjects = shuffled_subjects[:train_size]
        val_subjects = shuffled_subjects[train_size:]
        print(f"Training subjects: {len(train_subjects)}")  # 185
        print(f"Validation subjects: {len(val_subjects)}")  # 21
        
        # Split dataframe based on subjects
        train_df = df[df['Subject'].isin(train_subjects)] 
        val_df = df[df['Subject'].isin(val_subjects)]
        print(f"Training samples: {len(train_df)}")            # 630
        print(f"Validation samples: {len(val_df)}")            # 60

        train_list = train_df.values.tolist()
        val_list = val_df.values.tolist()

        # Save to pickle files
        with open(train_path, 'wb') as f:
            pickle.dump(train_list, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_list, f)
        print("Datasets saved!")

    def __getitem__(self, idx):
        subject, group, gender, age, sMRI_path, fMRI_path = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = nib.load(fMRI_path)
            fmri_data = fmri_img.dataobj[1:, 10:-9, 1: ,]          # Shape: (91, 109, 91, 140) for (H, W, D, T) -> (90, 90, 90, 140)
            fMRI_tensor = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
            fMRI_tensor = torch.tensor(fMRI_tensor, dtype=torch.float32)      # (90, 90, 90) shape

            # group_encoded = torch.tensor(0 if group == 'CN' else 1 if group in ['EMCI', 'LMCI'] else 2 if group == 'AD' else -1)     # 0: CN, 1: EMCI/LMCI, 2: AD, -1: unknown
            group_encoded = torch.tensor(0 if group == 'CN' else 1 if group in ['AD'] else -1)     # 0: CN, 1: AD, -1: unknown
            gender_encoded = torch.tensor(0 if gender == 'F' else 1)
            age = torch.tensor(age)
            age_group = torch.tensor(0 if age < 68 else 1 if age > 80 else -1) # min 56, max 96, median 74. Quartile1 = 68, Quartile3 = 80.

            if age < 68 and age > 80:
                print("ERROR: age out of bounds")
            return subject, fMRI_tensor, group_encoded, gender_encoded, age, age_group
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
