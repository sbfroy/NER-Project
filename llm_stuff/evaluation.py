import torch
from torchmetrics import Precision, Recall, F1Score
from src.utils.metrics import SpanAcc
from src.utils.label_mapping_regplans import id_to_label

def evaluate(preds, labels):
    num_classes = len(id_to_label)
    precision_metric = Precision(task='multiclass', num_classes=num_classes, average='macro')
    recall_metric = Recall(task='multiclass', num_classes=num_classes, average='macro')
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro')
    span_acc = SpanAcc(id_to_label)

    preds_tensor = torch.tensor(preds, dtype=torch.int64)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    precision_score = precision_metric(preds_tensor, labels_tensor).item()
    recall_score = recall_metric(preds_tensor, labels_tensor).item()
    f1_score = f1_metric(preds_tensor, labels_tensor).item()

    span_acc.update(preds_tensor, labels_tensor)
    span_acc_score = span_acc.compute().item()
    span_acc.reset()

    return {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'span_acc': span_acc_score
    }
