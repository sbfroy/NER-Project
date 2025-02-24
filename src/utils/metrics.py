import torch
from torchmetrics import Metric

# Custom metric to compute F1-score for entity spans

class SpanF1(Metric):
    def __init__(self, id_to_label):
        super().__init__()
        self.id_to_label = id_to_label
    
        self.add_state('correct_spans', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('predicted_spans', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('gold_spans', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')

    def extract_spans(self, labels):
       
        spans = set()
        start, end = None, None

        for idx, label_id in enumerate(labels):
            label = self.id_to_label[label_id]

            if label == 'B-FELT':
                if start is not None: spans.add((start, end))
                start = idx
                end = idx

            elif label == 'I-FELT' and start is not None:
                # Continue span
                end = idx  

            else:
                if start is not None: spans.add((start, end))  # Save last span
                start, end = None, None

        if start is not None:
            spans.add((start, end))

        return spans # Set of (start, end) tuples

    def update(self, preds: torch.Tensor, targets: torch.Tensor):

        preds = preds.reshape(1, -1)
        targets = targets.reshape(1, -1)

        for pred, target in zip(preds, targets):
            pred_spans = self.extract_spans(pred)
            gold_spans = self.extract_spans(target)

            self.correct_spans += len(pred_spans & gold_spans) 
            self.predicted_spans += len(pred_spans)
            self.gold_spans += len(gold_spans)

    def compute(self):
        
        precision = self.correct_spans / self.predicted_spans if self.predicted_spans > 0 else torch.tensor(0.0)
        recall = self.correct_spans / self.gold_spans if self.gold_spans > 0 else torch.tensor(0.0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)

        return {
            'span_precision': precision,
            'span_recall': recall,
            'span_f1': f1
        }