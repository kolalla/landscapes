import numpy as np
import os
import shutil
import random
import pickle
import time

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Functions
def preprocess_resnet(batch_size=32, num_workers=4, train_transforms=[]):
    """Preprocess images for ResNet50"""

    train_transforms = [transforms.RandomResizedCrop(224)] + train_transforms + [
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join('data', x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    class_names = image_datasets['train'].classes

    return image_datasets, dataloaders, dataset_sizes, class_names

def preprocess_tabular():
    """Preprocess images for XGBoost"""
    tabular_datasets = {}

    for subset in ['train', 'val', 'test']:
        features = []
        labels = []
        for inputs, classes in dataloaders[subset]:
            batch_flat = inputs.numpy().reshape(inputs.shape[0], -1)
            features.append(batch_flat)
            labels.extend(classes.numpy())
        tabular_datasets[subset] = (np.vstack(features), np.array(labels))

    return tabular_datasets

if __name__ == '__main__':
    start = time.time()

    print('Creating validation folder and moving random train images to validation...')

    # set up vars
    train_dir = 'data/train'
    val_dir = 'data/val'
    num_val_samples = 3000
    num_val_samples_per_class = num_val_samples // len(os.listdir(train_dir))

    # create validation folder or clean out if it already exists
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    else:
        shutil.rmtree(val_dir)
        os.makedirs(val_dir)

    for label in os.listdir(train_dir):
        new_label_dir = os.path.join(val_dir, label)
        if not os.path.exists(new_label_dir):
            os.makedirs(new_label_dir)

    print(f'Folders in {val_dir}:', os.listdir(val_dir))

    # move random train images to validation
    for label in os.listdir(train_dir):
        train_label_dir = os.path.join(train_dir, label)
        val_label_dir = os.path.join(val_dir, label)

        imgs = os.listdir(train_label_dir)
        random.shuffle(imgs)

        for img in imgs[:num_val_samples_per_class]:
            source_path = os.path.join(train_label_dir, img)
            target_path = os.path.join(val_label_dir, img)
            shutil.move(source_path, target_path)

    for subset in ['train', 'val', 'test']:
        num_files = 0
        for folder in os.listdir(f'data/{subset}'):
            num_files += len(os.listdir(f'data/{subset}/{folder}'))
        print(f'Number of {subset} files: {num_files}')

    
    print('\nPreprocessing images for ResNet50...')
    image_datasets, dataloaders, dataset_sizes, class_names = preprocess_resnet()

    print(
        f'Class names: {class_names}'
        f'\nDataset sizes: {dataset_sizes}'
    )

    print('\nCreating flat, tabular dataset for XGBoost...')
    tabular_datasets = preprocess_tabular()

    print(
        f'Train features shape: {tabular_datasets["train"][0].shape}\n'
        f'Train labels shape: {tabular_datasets["train"][1].shape}\n'
        f'Val features shape: {tabular_datasets["val"][0].shape}\n'
        f'Val labels shape: {tabular_datasets["val"][1].shape}\n'
        f'Test features shape: {tabular_datasets["test"][0].shape}\n'
        f'Test labels shape: {tabular_datasets["test"][1].shape}\n'
    )

    # Save dataset metadata and class names
    torch.save({
        'image_datasets': image_datasets,
        'class_names': class_names,
        'dataset_sizes': dataset_sizes
    }, 'data/image_datasets.pth')

    # Save dataloaders
    torch.save(dataloaders, 'data/dataloaders.pth')

    # Tabular datasets
    with open('data/tabular_datasets.pkl', 'wb') as f:
        pickle.dump(tabular_datasets, f)


    end = time.time()
    print(f'Time elapsed: {(end-start)/60:.2f} minutes')