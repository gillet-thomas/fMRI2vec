import torch
import pickle
import pandas as pd
import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import time 

from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset
from monai.transforms import Compose, RandSpatialCrop, ToTensor


# ADNI dataset class
class ADNIDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.csv_path = config['adni_csv']
        self.dataset_path = config['adni_train_path'] if mode == 'train' else config['adni_val_path']

        self.transform = Compose([
            RandSpatialCrop(roi_size=(75, 75, 75), random_center=True, random_size=False),
            ToTensor()
        ])
        
        # self.generate_data(config['adni_train_path'], config['adni_val_path'])
        # self.generate_folds('./src/data/')
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        # keep only people with age <Q1 or >Q3
        # self.data = [sample for sample in self.data if sample[5] < 68 or sample[5] > 80]
        self.data = [sample for sample in self.data if sample[3] in ['AD', 'CN', 'LMCI', 'EMCI']]

        print(f"Dataset initialized: {len(self.data)} {mode} samples")
        
    def generate_data(self, train_path, val_path):
        # Load CSV data
        df = pd.read_csv(self.csv_path, usecols=['ID', 'Subject', 'Group', 'Sex', 'Age', 'Path_sMRI_brain', 'Path_fMRI_brain'])
        print(f"Total rows in CSV: {len(df)}")              # 690
        
        # Get unique subjects and their counts
        unique_subjects = df['Subject'].unique()
        n_subjects = len(unique_subjects)
        print(f"Total unique subjects: {n_subjects}")       # 206
        
        # Randomly shuffle and split subjects
        shuffled_subjects = np.random.permutation(unique_subjects)
        train_size = int(0.9 * n_subjects)  
        train_subjects = shuffled_subjects[:train_size]
        val_subjects = shuffled_subjects[train_size:]
        
        print(f"Training subjects: {len(train_subjects)}")  # 185
        print(f"Validation subjects: {len(val_subjects)}")  # 21
        
        # Split dataframe based on subjects
        train_df = df[df['Subject'].isin(train_subjects)] 
        val_df = df[df['Subject'].isin(val_subjects)]
        
        print(f"Training rows: {len(train_df)}")            # 630
        print(f"Validation rows: {len(val_df)}")            # 60

        train_samples, val_samples = [], []

        # Process training data
        print("Processing training data...")
        for row in tqdm(train_df.itertuples(index=False), total=len(train_df)):
            subject, group, sex, age, path_fmri = row.Subject, row.Group, row.Sex, row.Age, row.Path_fMRI_brain
            samples = self.process_subject_data(subject, path_fmri, group, sex, age)
            train_samples.extend(samples)
        
        # Process validation data
        print("Processing validation data...")
        for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):
            subject, group, sex, age, path_fmri = row.Subject, row.Group, row.Sex, row.Age, row.Path_fMRI_brain
            samples = self.process_subject_data(subject, path_fmri, group, sex, age)
            val_samples.extend(samples)
        
        print(f"Processed {len(train_samples)} train samples")          # 630
        print(f"Processed {len(val_samples)} validation samples")       # 60
        
        # Save to pickle files
        with open(train_path, 'wb') as f:
            pickle.dump(train_samples, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_samples, f)
        print("Datasets saved!")

    def generate_folds(self, base_path):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Gender', 'Age', 'Age_Group', 'Pain_Distraction_Group'])
        
        # Get unique subjects and their counts
        unique_subjects = df['Subject'].unique()
        n_subjects = len(unique_subjects)
        print(f"Total unique subjects: {n_subjects}")  # 178
        
        # Randomly shuffle subjects
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        # Implement 5-fold cross-validation
        k_folds = 5
        fold_size = n_subjects // k_folds
        
        # Create directories for each fold if they don't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Process each fold
        for fold in range(k_folds):
            print(f"\nProcessing fold {fold+1}/{k_folds}")
            
            # Calculate start and end indices for validation subjects in this fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_subjects
            
            # Split subjects for this fold
            val_subjects = shuffled_subjects[val_start:val_end]
            train_subjects = np.concatenate([
                shuffled_subjects[:val_start],
                shuffled_subjects[val_end:]
            ])
            
            print(f"Training subjects: {len(train_subjects)}")
            print(f"Validation subjects: {len(val_subjects)}")
            
            # Split dataframe based on subjects
            train_df = df[df['Subject'].isin(train_subjects)]
            val_df = df[df['Subject'].isin(val_subjects)]
            
            print(f"Training rows: {len(train_df)}")
            print(f"Validation rows: {len(val_df)}")
            
            train_samples, val_samples = [], []
            
            # Process training data
            print("Processing training data...")
            for row in tqdm(train_df.itertuples(index=False), total=len(train_df)):
                subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
                samples = self.process_subject_data(subject, fmri_path, gender, age, age_group, pain_group)
                train_samples.extend(samples)
            
            # Process validation data
            print("Processing validation data...")
            for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):
                subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
                samples = self.process_subject_data(subject, fmri_path, gender, age, age_group, pain_group)
                val_samples.extend(samples)
            
            print(f"Processed {len(train_samples)} train samples")
            print(f"Processed {len(val_samples)} validation samples")
            
            # Create fold directory
            fold_dir = os.path.join(base_path, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Save to pickle files
            train_path = os.path.join(fold_dir, 'train_data.pkl')
            val_path = os.path.join(fold_dir, 'val_data.pkl')
            
            with open(train_path, 'wb') as f:
                pickle.dump(train_samples, f)
            with open(val_path, 'wb') as f:
                pickle.dump(val_samples, f)
            
            print(f"Fold {fold+1} datasets saved!")
        
        print("\nAll folds processed and saved successfully!")
    
    def process_subject_data(self, subject, fmri_path, group, gender, age):
        """Process a single subject's fMRI data and return samples."""
        samples = []
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()    # (91, 109, 91, 140)

            for timepoint in range(fmri_data.shape[-1]):
                samples.append((subject, timepoint, fmri_path, group, gender, age))
                
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            
        return samples

    def __getitem__(self, idx):
        subject, timepoint, fmri_path, group, gender, age = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = nib.load(fmri_path)
            fmri_data = fmri_img.dataobj[1:, 10:-9, 1: , timepoint]          # Shape: (91, 109, 91, 146) -> (90, 90, 90)
            mri_tensor = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
            mri_tensor = torch.tensor(mri_tensor, dtype=torch.float32)      # (90, 90, 90) shape
            
            if self.transform:
                mri_tensor = mri_tensor.unsqueeze(0)
                # plt.imsave("mri_tensor0.png", mri_tensor.squeeze(0)[:,:, 45].numpy())
                mri_tensor = self.transform(mri_tensor).squeeze
                # plt.imsave("mri_tensor1.png", mri_tensor.squeeze(0)[:,:, 45].numpy())
                # time.sleep(5)

            group_encoded = torch.tensor(0 if group == 'CN' else 1 if group in ['EMCI', 'LMCI'] else 2 if group == 'AD' else -1)     # 0: CN, 1: EMCI/LMCI, 2: AD, -1: unknown
            gender_encoded = torch.tensor(0 if gender == 'F' else 1)
            age = torch.tensor(age)
            age_group = torch.tensor(0 if age < 68 else 1)      # min 56, max 96, median 74. Quartile1 = 68, Quartile3 = 80.

            # if age < 68 and age > 80:
            #     print("ERROR: age out of bounds")
            return subject, timepoint, mri_tensor, group_encoded, gender_encoded, age, age_group
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
