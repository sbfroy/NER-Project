import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

class BERT(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BERT, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        print(self.config)
        self.transformer = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, labels):
        # Calculates the loss when labels is provided.
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs



