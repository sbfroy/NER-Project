import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

# TODO: Implement a dropout layer if overfitting.

class TransformerModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, trust_remote_code=True)
        #print(self.config)
        self.transformer = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config, trust_remote_code=True)

    def forward(self, input_ids, attention_mask, labels):
        # Calculates the loss (cross-entropy) when labels is provided.
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs



