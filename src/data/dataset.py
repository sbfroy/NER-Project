import torch
from torch.utils.data import Dataset
from src.utils.label_mapping import label_to_id

class Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Extract words and labels from a sentence
        words = self.df.iloc[idx]['words']
        labels = self.df.iloc[idx]['labels']

        # Original labels to integers
        labels = [label_to_id[label] for label in labels]

        #tokenIDs
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Allign labels on the new subword tokens
        alligned_labels = self.allign_labels(labels, encoding['offset_mapping'])

        encoding.pop('offset_mapping') # Dont need it anymore

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(alligned_labels, dtype=torch.long)
        }

    def allign_labels(self, labels, offsets):

        offsets = offsets.squeeze(0) # Remove unnecessary dimension
        alligned_labels = []
        label_idx = 0

        for offset in offsets:
            if offset[0].item() == 0 and offset[1].item() != 0:
                alligned_labels.append(labels[label_idx])
                label_idx += 1
            else:
                alligned_labels.append(-100)

        return alligned_labels
