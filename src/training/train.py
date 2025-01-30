import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_dataset, val_dataset, optimizer, batch_size, num_epochs, device):

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(num_epochs):

        model.train() ## training mode ##

        train_loss = 0

        for batch in tqdm(train_loader):

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

        with torch.no_grad():
            for batch in tqdm(val_loader):

                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, masks, labels)
                loss = outputs.loss

                val_loss += loss.item()
        
        val_loss /= len(val_loader)
    
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
    
    print('Training says SUUIII!')

    return history