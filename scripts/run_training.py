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
import wandb

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = TransformerModel(
    model_name=config['model']['model_name'], 
    dropout=config['model']['dropout'],
    num_labels=len(label_to_id)
)

"""# Layer freezing
layers = ['encoder.layer.6', 'encoder.layer.7', 'encoder.layer.8', 'encoder.layer.9', 'encoder.layer.10', 'encoder.layer.11']

for name, param in model.transformer.base_model.named_parameters():
    if any(layer in name for layer in layers):
        param.requires_grad = True  # Fine-tune just some layers
    else:
        param.requires_grad = False 
"""

# Track of what layers are frozen
frozen_layers = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        frozen_layers.append(name)

if not frozen_layers:
    frozen_layers = 'None'

# Data loading
train_df = create_df(base_dir / 'data/my_data/regplans-train.conllu')
val_df = create_df(base_dir / 'data/my_data/regplans-dev.conllu')
 
train_dataset = Dataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = Dataset(val_df, tokenizer, config['data']['max_seq_len'])

optimizer = optim.AdamW(
    model.parameters(), 
    lr=config['training']['general']['learning_rate'],
    weight_decay=config['training']['general']['weight_decay']
)

# Store params
wandb.init(
    project='NER_Exploration',
    config={
        'model_name': config['model']['model_name'],
        'dropout': config['model']['dropout'],
        'learning_rate': config['training']['general']['learning_rate'],
        'batch_size': config['training']['general']['batch_size'],
        'num_epochs': config['training']['general']['num_epochs'],
        'frozen_layers': frozen_layers
    }
)

train_model(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    optimizer = optimizer,
    batch_size = config['training']['general']['batch_size'],
    num_epochs = config['training']['general']['num_epochs'],
    device = device,
    id_to_label=id_to_label,
    wandb_log=True
)

# torch.save(model.state_dict(), output_dir / 'transformer_model.pth')
