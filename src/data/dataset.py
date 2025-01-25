import torch
import torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Extract words and labels from a sentence
        words = self.df[idx]['words']
        labels = self.df[idx]['labels']

        #tokenIDs
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mappings=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = self.allign_labels(labels, encoding['offset_mapping'])

        encoding.pop('offset_mapping')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def allign_labels(self,  labels, offsets):

        alligned_labels = []

        for offset, label in zip(offsets, labels):
            if offset[0] == 0 and offset[1] != 0:
                alligned_labels.append(label)
            else:
                alligned_labels.append(-100)

        return alligned_labels



