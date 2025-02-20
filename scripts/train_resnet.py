# core
import os
import json
import time
import datetime
import matplotlib.pyplot as plt
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
# style
plt.style.use('dark_background')

"""Functions"""

def load_data():
    """Loads dataset and returns necessary components."""
    data = torch.load('data/image_datasets.pth', weights_only=False)
    class_names = data['class_names']
    dataset_sizes = data['dataset_sizes']
    dataloaders = torch.load('data/dataloaders.pth', weights_only=False)

    return dataloaders, dataset_sizes, class_names

def train_model(dataloaders, dataset_sizes, class_names, learning_rate, dropout, optimizer_class, num_epochs=10, print_progress=True, save_model=True):
    """Train the model with given hyperparameters and return validation accuracy."""

    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    # Modify last layer and move to device
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.fc.parameters(), lr=learning_rate)

    # Set up variables
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Train model
    for epoch in range(num_epochs):
        
        if print_progress:
            if epoch % (num_epochs // 10) == 0 or epoch == num_epochs - 1:
                print(f'\nEpoch {epoch + 1}/{num_epochs} - Time Elapsed: {(time.time() - start) / 60:.2f} minutes\n----------')
        
        # Repeat for training and validation
        for phase in ['train', 'val']:
            # Set model to training mode or evaluation mode
            model.train() if phase == 'train' else model.eval()
            running_corrects = 0
            running_loss = 0.0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # Transfer to device and zero gradients
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Capture loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

            if print_progress:
                if epoch % (num_epochs // 10) == 0 or epoch == num_epochs - 1:
                    print(f'{phase.title()} Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')

    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

    if save_model:
        model_name = f'resnet50_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.pt'
        torch.save(model, f'models/{model_name}')
        print(f'Model saved as {model_name}')

    return results

"""Main"""

if __name__ == '__main__':

    # Read in config file
    with open('scripts/config.json', 'r') as f:
        config = json.load(f)
    LEARNING_RATE = config['learning_rate']
    DROPOUT = config['dropout']
    OPTIMIZER = {'Adam': optim.Adam, 'SGD': optim.SGD}[config['optimizer']]
    NUM_EPOCHS = config['num_epochs']

    # Create directory for saving models and logs
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Load datasets
    dataloaders, dataset_sizes, class_names = load_data()
    print(
        f'Data Loaded...'
        f'\nClass names: {class_names}'
        f'\nDataset sizes: {dataset_sizes}'
    )

    # Train model
    results = train_model(
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        class_names=class_names,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
        optimizer_class=OPTIMIZER,
        num_epochs=NUM_EPOCHS
    )

    # Sort results
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    train_accuracies = results['train_accuracies']
    val_accuracies = results['val_accuracies']

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Validation')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot([x.detach().cpu().numpy() for x in train_accuracies], label='Train')
    axes[1].plot([x.detach().cpu().numpy() for x in val_accuracies], label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    plt.show(block=True)
