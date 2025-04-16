import os
import time
import torch
import wandb
import pickle
import datetime
import torch.nn as nn
from tqdm import tqdm

class Trainer():
    def __init__(self, config, model, dataset_train, dataset_val):
        self.config = config
        self.device = config['device']
        self.model = model.to(self.device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

        self.data = dataset_train
        self.val_data = dataset_val
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

        total_groups = 181 + 116
        AD_count = 116
        CN_count = 181
        # weight=torch.tensor([total_groups/CN_count, total_groups/MCI_count, total_groups/AD_count]

        self.scaler = torch.amp.GradScaler()       # for Automatic Mixed Precision
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True, patience=1, factor=0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        self.log_interval = len(self.dataloader) // 10  # Log every 10% of batches

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")          ## 114 batches of size 64
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")    ## 13 batches of size 64
        print("=" * 50)

    def run(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"./results/{timestamp}"
        os.mkdir(path)

        print(f"Running on device: {self.device}")
        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)
            # self.scheduler.step(self.val_loss)

            torch.save(self.model.state_dict(), f'{path}/model-e{epoch}.pth')
            print(f"MODEL SAVED to .{path}/model-e{epoch}.pth")
    
    def train(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()
        
        accumulation_step = 8 # 8 times batch_size 4 = 32

        for i, (subject, fMRI, group, gender, age, age_group) in enumerate(self.dataloader):
            fMRI, age_group = fMRI.to(self.device), age_group.to(self.device)  ## (batch_size, 64, 64, 48, 140) and (batch_size)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(fMRI)  # output is [batch_size, 3]
                loss = self.criterion(outputs, age_group)
                # torch.cuda.empty_cache()  # Clear memory     
           
            # self.optimizer.zero_grad(set_to_none=True) # Modestly improve performance
            self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == age_group).sum().item()
            total += age_group.size(0)  # returns batch size

            # Gradients accumulation
            if (i + 1) % accumulation_step == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Logging
            if i != 0 and i % self.log_interval == 0:
                avg_loss = round(running_loss / self.log_interval, 5)
                accuracy = round(correct / total, 5)
                lr = round(self.optimizer.param_groups[0]['lr'], 5)
                duration = time.time() - start_time

                print(f"epoch {epoch}\t| batch {i}/{len(self.dataloader)}\t| train_loss: {avg_loss:.5f}\t| train_accuracy: {accuracy:.5f}\t| learning_rate: {lr:.5f}\t| duration: {duration:.2f}s")
                wandb.log({"epoch": epoch, "batch": i, "train_loss": avg_loss, "train_accuracy": accuracy, "learning_rate": lr, "duration": duration})
                
                correct, total, running_loss = 0, 0, 0.0
                start_time = time.time()  # Reset timer for next iteration

    def validate(self, epoch):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for i, (subject, fMRI, group, gender, age, age_group) in enumerate(self.val_dataloader):
                fMRI, age_group = fMRI.to(self.device), age_group.to(self.device)  ## (batch_size, 64, 64, 48) and (batch_size)
                outputs = self.model(fMRI)
                loss = self.criterion(outputs, age_group)
                val_loss += loss.item()
                correct += (outputs.argmax(dim=1) == age_group).sum().item()
                total += age_group.size(0)  # returns the batch size
                
            avg_val_loss = round(val_loss / len(self.val_dataloader), 5)
            self.val_loss = avg_val_loss # for LR scheduler
            accuracy = round(correct / total, 5)
            print(f"[VALIDATION] epoch {epoch}\t| total_batch {i}\t| val_loss {avg_val_loss:.5f}\t| val_accuracy {accuracy:.5f}")
            wandb.log({"epoch": epoch, "val_loss": avg_val_loss, "val_accuracy": accuracy})
    
    def evaluate_samples(self):
        self.model.eval()  # Set model to evaluation mode
        print("=" * 50)
        print(f"Training set has {len(self.data)} samples and validation set has {len(self.val_data)} samples.")
        print("Training loader has", len(self.dataloader), "batches and validation loader has", len(self.val_dataloader), "batches.")

        # Count number of unique subjects in training set
        with open(self.data.dataset_path, 'rb') as f: train_data = pickle.load(f)       # 78820 samples
        unique_train_subjects = list(set([sample[0] for sample in train_data]))         # 160 unique subjects
        # print(f"Unique training subjects: {len(unique_train_subjects)}")

        with open(self.val_data.dataset_path, 'rb') as f: val_data = pickle.load(f)     # 8680 samples
        unique_val_subjects = list(set([sample[0] for sample in val_data]))             # 17 unique subjects
        # print(f"Unique validation subjects: {len(unique_val_subjects)}")

        common = list(set(unique_train_subjects) & set(unique_val_subjects))
        print(f"Common subjects: {common}")                                             # 0 common subject  

        # Create evaluation dataset and dataloader
        evaluation_data = self.val_data
        # evaluation_data = torch.utils.data.Subset(self.val_data, range(100))
        evaluation_dataloader = torch.utils.data.DataLoader(evaluation_data, batch_size=1, shuffle=False, num_workers=self.num_workers)

        accuracy, duplicates = 0, 0
        with torch.no_grad():
            for i, (subject, fMRI, group, gender, age, age_group) in tqdm(enumerate(evaluation_dataloader), total=len(evaluation_dataloader)):
                subject = subject[0]
                fMRI = fMRI.to(self.device)
                predictions = self.model(fMRI)  # Get model predictions (batch_size, 4)

                prediction = predictions.argmax(dim=1).item()
                actual = group.item()
                # print(f"Predictions of {i}: {self.data.selected_groups[prediction]}/{self.data.selected_groups[actual]}")

                if subject in unique_train_subjects:
                    duplicates += 1
                    # print(f"Duplicate subject found: {subject}")

                accuracy += prediction == actual

        print(f"Accuracy: {accuracy/len(evaluation_dataloader)*100:.2f}%")
        print(f"Duplicates: {duplicates}")
