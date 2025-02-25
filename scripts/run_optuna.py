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
from src.utils.label_mapping_regplans import label_to_id, id_to_label
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
    learning_rate = trial.suggest_float('learning_rate', 3e-6, 1e-4, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=4)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.2, log=True)

    model = TransformerModel(
        model_name=config['model']['model_name'], 
        dropout=dropout,
        num_labels=len(label_to_id)
    )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_token_f1 = train_model(
        model = model,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        optimizer = optimizer,
        batch_size = batch_size,
        num_epochs = config['training']['general']['num_epochs'],
        device = device,
        id_to_label = id_to_label,
        trial=trial,
        verbose=False
    )

    return best_token_f1

# Optuna study
study = optuna.create_study(
    study_name='nb-bert-base',
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=config['general']['seed']),
    load_if_exists=True, 
    storage='sqlite:///nb-bert-base.db'
)

study.optimize(objective, n_trials=50, show_progress_bar=True)
