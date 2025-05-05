import sys
import os
import json
import random
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.utils.config_loader import load_config
from src.utils.seed import seed_everything
from src.data.preprocessing import create_df
from src.utils.label_mapping_regplans import label_to_id
from llm_stuff.evaluation import evaluate

# Append the parent directory to sys.path if needed for local module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def format_examples(example_subset):
    """
    Formats a subset of examples into a string for later prompt inclusion.
    """
    formatted = []
    for i, ex in enumerate(example_subset):
        # Join each token-label pair with a comma and a space
        entity_line = ", ".join([f"{e['word']} {e['label']}" for e in ex["entities"]])
        formatted.append(
            f"Eksempel {i+1}:\n"
            f"Tekst: \"{ex['sentence']}\"\n"
            f"Entiteter: {entity_line}\n"
            "##\n"
        )
    return "\n".join(formatted)

def main():
    base_dir = Path(os.getcwd())
    config = load_config(base_dir / 'secrets.yaml')
    seed_everything(42)

    val_df = create_df(base_dir / 'data/my_data/regplans-dev.conllu')

    with open(base_dir / 'llm_stuff/prompts/examples.json', 'r', encoding='utf-8') as f:
        example_bank = json.load(f)

    ids = [1, 19, 16, 3, 21]
    examples = [next(ex for ex in example_bank if ex["id"] == id) for id in ids]
    formatted_examples = format_examples(examples)
    print(formatted_examples)

    model_name = 'flan-t5-xl'
    model_path = f'google/{model_name}'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    all_pred_ids = []
    all_true_ids = []
    all_results = []

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        sentence = row['full_text']
        tokens   = row['words']
        true_labels = row['labels']

        input_text = f"""
        Du er en ekspert på Named Entity Recognition (NER). Din oppgave er å identifisere feltnavn entiteter.

        Gyldige etiketter er B-FELT (starten på et feltnavn) og I-FELT (fortsettelsen).

        {formatted_examples}

        Separer token-etikett par med ett komma. Hvert par må inkludere tokenet og etiketten, atskilt med ett mellomrom.

        Tekst: '{sentence}'

        Entiteter:
        """

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs   = model.generate(input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        entities = defaultdict(list)
        for pair in generated_text.split(','):
            parts = pair.strip().split()
            if len(parts) == 2:
                word, label = parts
                entities[word].append(label)

        pred_labels = []
        word_counts = defaultdict(int)
        for token in tokens:
            if token in entities and word_counts[token] < len(entities[token]):
                pred_labels.append(entities[token][word_counts[token]])
                word_counts[token] += 1
            else:
                pred_labels.append("O")

        pred_ids = [
            label_to_id[label] if label in label_to_id else label_to_id.get("O", -1)
            for label in pred_labels
        ]
        true_ids = [label_to_id[label] for label in true_labels]

        all_pred_ids.extend(pred_ids)
        all_true_ids.extend(true_ids)

        all_results.append({
            'sentence': sentence,
            'tokens': tokens,
            'true_labels': true_labels,
            'predicted_labels': pred_labels,
            'generated_text': generated_text,
        })

    metrics = evaluate(all_true_ids, all_pred_ids)
    print("Evaluation Metrics on Val Set:")
    print(metrics)

    final_output = {
        'prompt': input_text,
        'evaluation_metrics': metrics,
        'results': all_results
    }

    output_file = base_dir / f"llm_stuff/results/{model_name}_FEWSHOT_5_FULLVALSET.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
