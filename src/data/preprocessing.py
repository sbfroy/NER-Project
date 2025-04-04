import pandas as pd
from conllu import parse
from pathlib import Path

base_dir = Path(__file__).parent.parent.parent

def create_df(file_path):

    """
    Reads a CONLL-U formatted file, extracts words and enitity labels, 
    and returns a df with the words and labels for each sentence.
    """

    # Open and read the CoNLL-U file
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = parse(f.read())

    data = {'full_text': [], 'words': [], 'labels': []}

    for sentence in sentences:

        # TODO: Add some error handling if missing data

        full_text = sentence.metadata['text'] # Extract the full text
        words = [word['form'] for word in sentence] # Extract the words
        labels = [word['misc']['name'] for word in sentence] # Extract the entity labels

        data['full_text'].append(full_text)
        data['words'].append(words)
        data['labels'].append(labels)

    return pd.DataFrame(data)
