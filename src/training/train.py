import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.utils.label_mapping import id_to_label
from tqdm import tqdm
import optuna

def train_model(model, train_dataset, val_dataset, optimizer, batch_size, num_epochs, device, trial=None,
                verbose=True):

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'class_report': []
    }

    for epoch in range(num_epochs):

        model.train() ## training mode ##

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

        model.eval() ## evaluation mode ##

        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='val', leave=False, ncols=75, disable=not verbose):

                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, masks, labels)
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1) # Get the most likely entity class idx
                preds = preds.view(-1).cpu().numpy() # Flatten the tensor
                labels = labels.view(-1).cpu().numpy()

                # Filter out -100 tokens
                preds = preds[labels != -100]
                labels = labels[labels != -100]

                all_preds.extend(preds)
                all_labels.extend(labels)

                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # ids to labels
        all_labels = [id_to_label[label] for label in all_labels] 
        all_preds = [id_to_label[pred] for pred in all_preds] 

        class_report = classification_report(
            all_labels, all_preds, zero_division=0, digits=4, output_dict=True
        )
        history['class_report'].append(class_report)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')

    if verbose:       
        print("\n Final classification report:")
        print(classification_report(all_labels, all_preds, zero_division=0, digits=4))

        print('Training says SUUIII!')

    # Optuna
    if trial:
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return history
