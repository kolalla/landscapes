# core
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
# metrics
from sklearn.metrics import classification_report
# custom
from train_resnet import load_data
# style
plt.style.use('dark_background')

def analyze_errors(model_name, dataloaders, class_names):
    """Generate classification reports and plot predictions by class."""
    # Set device and model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f'models/{model_name}', weights_only=False)
    model = model.to(device)

    ### TRAINING ERRORS
    true_labels = []
    pred_labels = []

    # Loop through validation set
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Generate predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Store true and predicted labels
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Print classification report
    print('\nClassification Report -- Train\n')
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    ### VALIDATION ERRORS
    true_labels = []
    pred_labels = []

    # Loop through validation set
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Generate predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Store true and predicted labels
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Print classification report
    print('\nClassification Report -- Validation\n')
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # Plot predictions by class
    print('\nPredictions by Class -- Validation\n')
    error_df = pd.DataFrame({'true_label': true_labels, 'pred_label': pred_labels})
    error_df = error_df[true_labels != pred_labels].reset_index(drop=True)
    error_df.true_label = error_df.apply(lambda row: class_names[row['true_label']], axis=1)
    error_df.pred_label = error_df.apply(lambda row: class_names[row['pred_label']], axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
    for label, ax in zip(class_names, axes.ravel()):
        label_df = error_df[error_df.true_label == label]
        ax.bar(label_df.pred_label.value_counts().index, label_df.pred_label.value_counts().values)
        ax.set_title(label.title())
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Count')
    plt.show(block=True)

if __name__ == '__main__':
    # Load datasets
    dataloaders, dataset_sizes, class_names = load_data()
    print(
        f'Data Loaded...'
        f'\nClass names: {class_names}'
        f'\nDataset sizes: {dataset_sizes}'
    )
    # Load model and analyze errors
    latest_model_name = 'resnet50_20250220_0939.pt' # os.listdir('models')[0]
    analyze_errors(latest_model_name, dataloaders, class_names)