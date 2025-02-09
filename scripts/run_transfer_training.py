import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from transformers import AutoTokenizer
from src.data.dataset import Dataset
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config
from src.utils.label_mapping_regplan import label_to_id
from src.utils.seed import seed_everything
from src.training.train import train_model
from src.models.transformer_model import TransformerModel

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

pretrained = torch.load(base_dir / 'src/models/transformer_model.pth', map_location=device)
pretrained = {k: v for k, v in pretrained.items() if 'classifier' not in k} # Remove classification head  

model.load_state_dict(pretrained, strict=False)
model.transformer.classifier = nn.Linear(model.config.hidden_size, len(label_to_id)).to(device) # Reinitialize classification head

layers = ['encoder.layer.6', 'encoder.layer.7', 'encoder.layer.8', 'encoder.layer.9', 'encoder.layer.10', 'encoder.layer.11']

for name, param in model.transformer.base_model.named_parameters():
    if any(layer in name for layer in layers):
        param.requires_grad = True  # Fine-tuning just some layers
    else:
        param.requires_grad = False 

for name, param in model.named_parameters():
    # Print what layers are trainable
    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

# Data loading
train_df = create_df(base_dir / 'data/my_data/regplan-train.conllu')
val_df = create_df(base_dir / 'data/my_data/regplan-dev.conllu')
 
train_dataset = Dataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = Dataset(val_df, tokenizer, config['data']['max_seq_len'])

optimizer = optim.AdamW(model.parameters(), lr=config['training']['transfer']['learning_rate'])

train_model(
    model = model,
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    optimizer = optimizer,
    batch_size = config['training']['transfer']['batch_size'],
    num_epochs = config['training']['transfer']['num_epochs'],
    device = device
)

torch.save(model.state_dict(), 'FINETUNED_ON_NER_transformer_model.pth')
