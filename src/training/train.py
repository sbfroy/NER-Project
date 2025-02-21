import torch
from torchmetrics import F1Score, Precision, Recall
from src.utils.metrics import ExactMatchAccuracy
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
import wandb

def train_model(model, train_dataset, val_dataset, optimizer, batch_size, num_epochs, device, id_to_label, 
                trial=None, verbose=True, wandb_log=False):
    
    # Initialize torchmetrics
    f1 = F1Score(task='multiclass', num_classes=len(id_to_label), average='macro').to(device)
    precision = Precision(task='multiclass', num_classes=len(id_to_label), average='macro').to(device)
    recall = Recall(task='multiclass', num_classes=len(id_to_label), average='macro').to(device)
    exact_match_acc = ExactMatchAccuracy().to(device)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    best_f1 = 0

    for epoch in range(num_epochs):

        model.train() # Training mode
        train_loss = 0

        for batch in tqdm(train_loader, desc='train', leave=False, ncols=75, disable=not verbose):

            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks, labels) # Forward pass
            loss = outputs.loss

            loss.backward()
            optimizer.step()  

            train_loss += loss.item()
        
        train_loss /= len(train_loader) 

        model.eval() # Evaluation mode
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc='val', leave=False, ncols=75, disable=not verbose):

                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, masks, labels)
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1) # Most likely entity class idx
                preds = preds.view(-1).cpu().numpy() # Flatten the tensor
                labels = labels.view(-1).cpu().numpy()

                # Filter out -100 tokens
                preds = preds[labels != -100]
                labels = labels[labels != -100]

                all_preds.extend(preds)
                all_labels.extend(labels)

                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        exact_match_acc.update(torch.tensor(all_preds, device=device), torch.tensor(all_labels, device=device))
        f1_score = f1(torch.tensor(all_preds, device=device), torch.tensor(all_labels, device=device)).item()
        precision_score = precision(torch.tensor(all_preds, device=device), torch.tensor(all_labels, device=device)).item()
        recall_score = recall(torch.tensor(all_preds, device=device), torch.tensor(all_labels, device=device)).item()
        exact_match_score = exact_match_acc.compute().item()

        # ids to labels
        all_labels = [id_to_label[label] for label in all_labels] 
        all_preds = [id_to_label[pred] for pred in all_preds] 

        class_report_dict = classification_report(all_labels, all_preds, zero_division=0, digits=4, output_dict=True)

        if f1_score > best_f1:
            best_f1 = f1_score

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
                f'F1-score: {f1_score:.4f}, Precision: {precision_score:.4f}, Recall: {recall_score:.4f}, '
                f'Exact match accuracy: {exact_match_score:.4f}')
    
        if wandb_log:
            # Logs classification report as a table
            table = wandb.Table(columns=['Entity', 'Precision', 'Recall', 'F1-Score', 'Support'])
            for entity, metrics in class_report_dict.items():
                if entity in {'accuracy', 'macro avg', 'weighted avg'}:
                    continue
                table.add_data(entity, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support'])

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'f1-score': f1_score,
                'precision': precision_score,
                'recall': recall_score,
                'exact_match_accuracy': exact_match_score,
                'classification_report': table
            })

        # Optuna
        if trial:
            trial.report(f1_score, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    print('Training says SUUIII!')
    wandb.finish()
    return best_f1 