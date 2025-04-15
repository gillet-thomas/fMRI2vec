import sys
import yaml
import wandb
import torch
import argparse
import warnings
import numpy as np
from src.data.DatasetADNI_4D import ADNIDataset4D   
from src.fmriEncoder import fmriEncoder
from src.Trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate fMRI Model")
    parser.add_argument("name", type=str, nargs="?", default=None, help="WandB run name (optional)")
    parser.add_argument('--inference', action='store_true', help='Run in inference mode')
    parser.add_argument('--sweep', action='store_true', help='Run WandB sweep')
    parser.add_argument('--cuda', type=int, default=2, help='CUDA device to use (e.g., 0 for GPU 0)')
    parser.add_argument('--wandb', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable Weights and Biases (WandB) tracking')
    return parser.parse_args()

def get_device(cuda_device):
    return f'cuda:{cuda_device}' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_config(args):
    config = yaml.safe_load(open("./configs/config.yaml"))
    config["device"] = get_device(args.cuda)
    config.update({"wandb_enabled": args.wandb, "name": args.name, "inference": args.inference, "sweep": args.sweep})
    return config

def train_sweep():
    # This function will be called by wandb.agent for each sweep run
    with wandb.init() as run:
        # Get config
        args = parse_args()
        base_config = get_config(args)      # Load base config (config.yaml) with args values
        sweep_config = wandb.config         # Load sweep config used in wandb initialization
        base_config.update(sweep_config)    # Overwrite base config with sweep config parameters values

        # Run training
        set_seeds(base_config)
        dataset_train, dataset_val = get_datasets(base_config)
        model = fmriEncoder(base_config)
        trainer = Trainer(base_config, model, dataset_train, dataset_val)
        trainer.run()

def set_seeds(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

def get_datasets(config):
    dataset_train = ADNIDataset4D(config, mode="train")
    dataset_val = ADNIDataset4D(config, mode="val")
    return dataset_train, dataset_val

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Parse arguments and load config
    args = parse_args()
    config = get_config(args)

    if not config["inference"] and not config["sweep"]:
        print("Training mode enabled.")

        # for fold in range(0, 5):
        #     print(f"FOLD {fold}/5 training...")
        #     fold_id = fold + 1
        #     config["dataset_train_path"] = './src/data/fold_' + str(fold_id) + '/train_data.pkl'
        #     config["dataset_val_path"] = './src/data/fold_' + str(fold_id) + '/val_data.pkl'

        wandb.init(project="fMRI2Vec", mode='online' if config["wandb_enabled"] else 'disabled', config=config, name=config["name"]) 
        set_seeds(config)
        dataset_train, dataset_val = get_datasets(config)
        model = fmriEncoder(config)
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.run()

            # print(f"FOLD {fold}/5 completed.")
            # print("=" * 50)

    elif not config["inference"] and config["sweep"]:
        print("Sweep mode enabled.")
        sweep_config = yaml.safe_load(open("./configs/sweep.yaml"))     # Load sweep configuration
        sweep_id = wandb.sweep(sweep_config, project="fMRI2Vec_Sweep")  # Initialize sweep
        wandb.agent(sweep_id, function=train_sweep, count=50)            # Start the sweep agent

    elif config["inference"] is True:
        print("Training is disabled. Inference only.")
        dataset_train, dataset_val = get_datasets(config)
        model = fmriEncoder(config)
        model.load_state_dict(torch.load(config['best_model_path'], map_location=config["device"], weights_only=True))
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.evaluate_samples()

if __name__ == "__main__":
    main()