import pandas as pd
import time
import datetime
import optuna
import torch.optim as optim
from torchvision import transforms
from preprocess import preprocess_resnet
from train_resnet import train_model # , load_data

def compute_ewa(values, alpha=0.3):
    """Computes an exponentially weighted average (EWA) for a list of values."""
    ewa = 0
    for val in values:
        ewa = alpha * val + (1 - alpha) * ewa
    return ewa

# Define objective function
def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    # Preprocessing hyperparameters
    batch_size = trial.suggest_int('batch_size', 8, 64)
    hortizontal_flip = trial.suggest_categorical('random_horizontal_flip', [True, False])
    rotation = trial.suggest_categorical('random_rotation', [True, False])
    color_jitter = trial.suggest_categorical('random_color_jitter', [True, False])
    transforms_dict = {
        'horizontal_flip': (hortizontal_flip, transforms.RandomHorizontalFlip()),
        'rotation': (rotation, transforms.RandomRotation(15)),
        'color_jitter': (color_jitter, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    }
    train_transforms = [x[1] for x in transforms_dict.values() if x[0]]

    # Preprocess data
    _, dataloaders, dataset_sizes, class_names = preprocess_resnet(
        batch_size=batch_size, num_workers=1, train_transforms=train_transforms
    )
    
    # Training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    optimizer_class = {'Adam': optim.Adam, 'SGD': optim.SGD}[optimizer_name]
    num_epochs = trial.suggest_int('num_epochs', 10, 200)

    # Train model
    start = time.time()
    results = train_model(
        dataloaders, dataset_sizes, class_names, 
        learning_rate, dropout, optimizer_class, num_epochs,
        print_progress=False, save_model=False
    )
    val_accuracy = results['val_accuracies'][-1]

    # Log results
    trial.set_user_attr('train_losses', results['train_losses'])
    trial.set_user_attr('val_losses', results['val_losses'])
    trial.set_user_attr('train_accuracies', results['train_accuracies'])
    trial.set_user_attr('val_accuracies', results['val_accuracies'])
    trial.set_user_attr('time_elapsed', (time.time() - start) / 60)

    # Return validation accuracy as the objective value
    return val_accuracy

if __name__ == '__main__':    
    # Run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    # Log best hyperparameters
    log_filename = f'logs/tuning-{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.csv'
    df = pd.DataFrame(study.trials_dataframe())
    df.to_csv(log_filename)