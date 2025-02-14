import torch
from transformers import AutoTokenizer
from src.models.transformer_model import TransformerModel
from src.utils.config_loader import load_config
from src.utils.label_mapping_transfer import label_to_id, id_to_label
from pathlib import Path
from tqdm import tqdm
import spacy
import PyPDF2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_text(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(f'File not found: {pdf_path}')
    
    text = ''
    with open(pdf_path, 'rb') as file:  
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'

    return text

def get_sent_tokens(text, nlp):
    if not text:
        return []
    
    doc = nlp(text)
    sent_tokens = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_tokens.append([token.text for token in sent])

    return sent_tokens

def get_predictions(sent_tokens, model, tokenizer):

    final_preds = []

    for sent in tqdm(sent_tokens):
        if not sent:
            continue

        encoding = tokenizer(
            sent,
            is_split_into_words=True,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors='pt',
            padding='max_length',
            max_length=config['data']['max_seq_len']
        )

        inputs = encoding['input_ids'].to(device)
        masks = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs, masks, labels=None)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs.squeeze(0))
        pred_labels = [id_to_label.get(pred, 'O') for pred in preds] 

        aligned_preds = []
        aligned_words = []
        word_idx = -1

        offsets = encoding['offset_mapping'].squeeze(0)

        # Align predictions with words
        for j, (token, offset) in enumerate(zip(tokens, offsets)):
            if offset[0] == 0 and offset[1] != 0: # New word
                word_idx += 1
                if word_idx < len(sent):
                    aligned_words.append(sent[word_idx])
                    aligned_preds.append(pred_labels[j])

        # Only keep 'B-FELT' and 'I-FELT'
        felt_tokens = [word for word, label in zip(aligned_words, aligned_preds) if label in ['B-FELT', 'I-FELT']]
        
        # Combine 'B-FELT' and 'I-FELT' tokens
        combined_tokens = []
        for i, token in enumerate(felt_tokens):
            if i == 0 or felt_tokens[i-1] == 'B-FELT':
                combined_tokens.append(token)
            else:
                combined_tokens[-1] += ' ' + token
        
        final_preds.extend(combined_tokens)

        # TODO: Make a regex function that show all zones in 'BKS1-BKS6'
            
    return list(dict.fromkeys(final_preds)) # Only return unique tokens

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

nlp = spacy.load('nb_core_news_sm')

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

model = TransformerModel(
    model_name=config['model']['model_name'], 
    dropout=config['model']['dropout'],
    num_labels=len(label_to_id)
)

model_path = base_dir / 'src' /'models' / 'transfer_transformer_model.pth'
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device)) 
model.to(device)
model.eval()

pdf_path = base_dir / 'data/pdfs/4219_201801_Bestemmelser_Solstad_221027.pdf'

text = get_text(pdf_path) # Get text from pdf
sent_tokens = get_sent_tokens(text, nlp) # Get tokens per sentence

preds = get_predictions(sent_tokens, model, tokenizer) 