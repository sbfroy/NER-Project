import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from tqdm import tqdm

def train_model(model, train_dataset, val_dataset, optimizer, batch_size, num_epochs, device):

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    for epoch in range(num_epochs):

        model.train() ## training mode ##

        train_loss = 0

        for batch in tqdm(train_loader, desc='train', leave=False, ncols=75):

            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader) 

        model.eval() ## evaluation mode ##

        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='val', leave=False, ncols=75):

                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, masks, labels)
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)
                preds = preds.view(-1).cpu().numpy()
                labels = labels.view(-1).cpu().numpy()

                valid_indices = labels != -100
                preds = preds[valid_indices]
                labels = labels[valid_indices]

                all_preds.extend(preds)
                all_labels.extend(labels)

                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=np.nan)

        # TODO: Make the metrics better

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1_score'].append(f1)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
              f'Acc: {accuracy:.4f}, Prec: {precision:.4f}, '
              f'Rec: {recall:.4f}, F1: {f1:.4f}')
        
        print("\n Classification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0, digits=4))
    
    print('Training says SUUIII!')

    return history
