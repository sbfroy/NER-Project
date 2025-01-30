import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from transformers import AutoTokenizer
from src.data.dataset import Dataset
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config
from src.utils.label_mapping import label_to_id
from src.utils.seed import seed_everything
from src.models.BERT_model import BERT
from src.training.train import train_model

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = BERT(model_name=config['model']['model_name'], num_labels=len(label_to_id))

# Data loading
train_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-train.conllu')
val_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-dev.conllu')
 
train_dataset = Dataset(train_df, tokenizer, 69)
val_dataset = Dataset(val_df, tokenizer, 69)

optimizer = optim.AdamW(model.parameters(), lr=3e-5)

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
