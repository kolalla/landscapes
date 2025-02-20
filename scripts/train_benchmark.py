#core
import numpy as np
import pickle
import time

# data 
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# modeling
import xgboost as xgb
from xgboost import XGBClassifier

start = time.time()

# Load data
with open('data/tabular_datasets.pkl', 'rb') as f:
    tabular_datasets = pickle.load(f)

image_datasets = torch.load('data/image_datasets.pth', weights_only=False)
class_names = image_datasets['class_names']

# Perform PCA to generate features
pca_datasets = {}
pca = PCA(n_components=1000)

for subset in ['train', 'val', 'test']:
    pca.fit(tabular_datasets[subset][0])
    pca_datasets.update({subset: (
        pca.transform(tabular_datasets[subset][0]), 
        tabular_datasets[subset][1]
    )})

print(
    f'Train PCA shape: {pca_datasets["train"][0].shape}\n'
    f'Val PCA shape: {pca_datasets["val"][0].shape}\n'
    f'Test PCA shape: {pca_datasets["test"][0].shape}'
)

# Set up pca data
X_train_pca, y_train = pca_datasets['train']
X_val_pca, y_val = pca_datasets['val']

# Create GPU-compatible DMatrix
dtrain = xgb.QuantileDMatrix(X_train_pca, label=y_train)
dval = xgb.QuantileDMatrix(X_val_pca, label=y_val)

# Train the model
xgb_pca = XGBClassifier(tree_method='hist', device='cuda', objective='multi:softprob', num_class=len(np.unique(y_train)))
xgb_pca.fit(X_train_pca, y_train)

# Evaluate the model on validation set
y_pred_pca = xgb_pca.predict(X_val_pca)
print(classification_report(y_val, y_pred_pca, target_names=class_names))

end = time.time()
print(f'Time elapsed: {(end-start)/60:.2f} minutes')