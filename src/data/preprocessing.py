from conllu import parse
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.parent.parent

def create_df(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = parse(f.read())

    data = {'words': [], 'labels': []}

    for sentence in sentences:

        words = [word['form'] for word in sentence]
        labels = [word['misc']['name'] for word in sentence]

        data['words'].append(words)
        data['labels'].append(labels)

    return pd.DataFrame(data)


file_path = base_dir /'data/norne/ud/nob/no_bokmaal-ud-train.conllu'

df = create_df(file_path)

print(df.head())
