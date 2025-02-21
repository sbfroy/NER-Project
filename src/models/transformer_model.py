import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig, logging

logging.set_verbosity_error()

class TransformerModel(nn.Module):
    def __init__(self, model_name, dropout, num_labels):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.transformer = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)

        if hasattr(self.transformer, 'dropout'):
            # Input to classifiaction head
            self.transformer.dropout.p = dropout

    def forward(self, input_ids, attention_mask, labels):
        # Calculates the loss (cross-entropy) when labels is provided.
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
