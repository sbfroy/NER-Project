import torch
from transformers import AutoTokenizer
from src.models.transformer_model import TransformerModel
from src.utils.config_loader import load_config
from src.utils.label_mapping import label_to_id, id_to_label
from pathlib import Path
from pypdf import PdfReader

# TODO: Run a whole pdf on this

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = TransformerModel(
    model_name=config['model']['model_name'], 
    dropout=config['model']['dropout'],
    num_labels=len(label_to_id)
)

model.load_state_dict(torch.load(base_dir / 'src' /'models' / 'transformer_model.pth', map_location=device)) 
model.to(device)
model.eval()

def inference(sentence):

    encoding = tokenizer(sentence, 
                         return_tensors="pt", 
                         padding="max_length", 
                         truncation=True, 
                         max_length=config['data']['max_seq_len'])

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    predicted_labels = [id_to_label.get(pred, "O") for pred in predictions]  # "O" for unknown

    return list(zip(tokens, predicted_labels))

# Get text from pdf
pdf_text = []
reader = PdfReader(base_dir / 'data/pdfs/EH_detaljregulering_for_kjetså_massetak.pdf')
for page in reader.pages:
    text = page.extract_text()
    pdf_text.append(text)

pdf_text = " ".join(pdf_text)
all_sentences = pdf_text.split('.')

for sentence in all_sentences[100:120]:
    predictions = inference(sentence)

    print("\n Predictions: ")
    for token, label in predictions:
        print(f"{token} --> {label}")