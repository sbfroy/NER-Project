import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from src.data.dataset import Dataset
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config
from src.utils.label_mapping import label_to_id
from src.utils.seed import seed_everything
from src.models.transformer_model import TransformerModel
from src.training.train import train_model
from datetime import datetime
import yaml

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = TransformerModel(
    model_name=config['model']['model_name'], 
    dropout=config['model']['dropout'],
    num_labels=len(label_to_id)
)

# TODO: Should I freeze some layers here too? 

# Data loading
train_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-train.conllu')
val_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-dev.conllu')
 
train_dataset = Dataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = Dataset(val_df, tokenizer, config['data']['max_seq_len'])

optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

# TODO: Implement my own criterion

history = train_model(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    optimizer = optimizer,
    batch_size = config['training']['batch_size'],
    num_epochs = config['training']['num_epochs'],
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

# Store data in individual folders
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = base_dir / 'logs' / f'run_{timestamp}'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'params.yaml', 'w') as f:
    yaml.dump(config, f)

pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)

torch.save(model.state_dict(), output_dir / 'transformer_model.pth')
