import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from pathlib import Path
from transformers import AutoTokenizer
from src.data.dataset import Dataset
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config
from src.utils.label_mapping_transfer import label_to_id
from src.utils.seed import seed_everything
from src.models.transformer_model import TransformerModel
from src.training.train import train_model
import optuna

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

optuna.logging.set_verbosity(optuna.logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

# Data loading
train_df = create_df(base_dir / 'data/my_data/regplans-train.conllu')
val_df = create_df(base_dir / 'data/my_data/regplans-dev.conllu')
 
train_dataset = Dataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = Dataset(val_df, tokenizer, config['data']['max_seq_len'])

def objective(trial):

    # params to tune
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 3e-7, 5e-5, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)

    model = TransformerModel(
        model_name=config['model']['model_name'], 
        dropout=dropout,
        num_labels=len(label_to_id)
    )

    model_path = base_dir / 'src' /'models' / 'transformer_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device)) 
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    history = train_model(
        model = model,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        optimizer = optimizer,
        batch_size = batch_size,
        num_epochs = config['training']['transfer']['num_epochs'],
        device = device,
        trial=trial,
        verbose=False
    )

    best_val_loss = min(history['val_loss'])

    return best_val_loss

# Optuna study
study = optuna.create_study(
    study_name='transfer_study',
    direction='minimize', 
    sampler=optuna.samplers.TPESampler(seed=config['general']['seed']),
    load_if_exists=True, 
    storage='sqlite:///transfer_study.db'
)

study.optimize(objective, n_trials=100, show_progress_bar=True)
