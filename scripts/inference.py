import torch
from transformers import AutoTokenizer
from src.models.BERT_model import BERT
from src.utils.config_loader import load_config
from src.utils.label_mapping import label_to_id, id_to_label
from pathlib import Path

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = BERT(model_name=config['model']['model_name'], num_labels=len(label_to_id))

model.load_state_dict(torch.load(base_dir / 'src' /'models' / 'transformer_model.pth', map_location=device)) 
model.to(device)
model.eval()

def inference(sentence):

    encoding = tokenizer(sentence, 
                         return_tensors="pt", 
                         padding="max_length", 
                         truncation=True, 
                         max_length=69)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)


    predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    predicted_labels = [id_to_label.get(pred, "O") for pred in predictions]  # "O" for unknown

    return list(zip(tokens, predicted_labels))


sentence = 'Ole bor i Kristiansand, og Are bor i Vennesla'

predictions = inference(sentence)

print("\nPredictions:")
for token, label in predictions:
    print(f"{token} --> {label}")