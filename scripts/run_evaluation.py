import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from src.data.dataset import Dataset
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config
from src.utils.label_mapping import label_to_id, id_to_label
from src.utils.seed import seed_everything
from src.models.transformer_model import TransformerModel
from sklearn.metrics import classification_report
from tqdm import tqdm

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything(config['general']['seed'])

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = TransformerModel(
    model_name=config['model']['model_name'], 
    dropout=config['model']['dropout'],
    num_labels=len(label_to_id)
)

# Data loading
test_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-test.conllu')
test_dataset = Dataset(test_df, tokenizer, config['data']['max_seq_len'])
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

model.load_state_dict(torch.load(base_dir / 'src/models/transformer_model.pth', map_location=device)) 
model.to(device)
model.eval()

test_loss = 0
test_preds = []
test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='test', leave=False, ncols=75):
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs, masks, labels)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1) 
        preds = preds.view(-1).cpu().numpy() 
        labels = labels.view(-1).cpu().numpy()

        # Filter out -100 tokens
        preds = preds[labels != -100]
        labels = labels[labels != -100]

        test_preds.extend(preds)
        test_labels.extend(labels)

        test_loss += loss.item()
        
    test_loss /= len(test_loader)

    # ids to labels
    test_labels = [id_to_label[label] for label in test_labels] 
    test_preds = [id_to_label[pred] for pred in test_preds] 

print('Final loss: ', test_loss)

print("Final classification report:")
print(classification_report(test_labels, test_preds, zero_division=0, digits=4))
