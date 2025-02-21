import torch
from torchmetrics import Metric

class ExactMatchAccuracy(Metric):
    # Computes how often the model predicts the WHOLE entity

    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        preds, labels = preds.tolist(), labels.tolist()
        
        # + 1 if there is a match between pred and label
        correct_matches = sum(1 for p, t in zip(preds, labels) if p == t)

        self.correct += correct_matches
        self.total += len(labels)

    def compute(self):
        exact_match_accuracy = self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)
        return exact_match_accuracy
