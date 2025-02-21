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

start = time.time()

# Load data
with open('data/tabular_datasets.pkl', 'rb') as f:
    tabular_datasets = pickle.load(f)

image_datasets = torch.load('data/image_datasets.pth', weights_only=False)
class_names = image_datasets['class_names']

# Perform PCA to generate features
n_components = int(tabular_datasets['train'][1].shape[0]/2)
pca = PCA(n_components=n_components)
print(f'Performing PCA with {n_components} components...\n')

# Set up pca data
X_train = pca.fit_transform(tabular_datasets['train'][0])
y_train = tabular_datasets['train'][1]
X_val = pca.transform(tabular_datasets['val'][0])
y_val = tabular_datasets['val'][1]

print(
    f'PCA Complete - Time elapsed: {(time.time()-start)/60:.2f} minutes\n'
    f'Train PCA shape: {X_train.shape}\n'
    f'Val PCA shape: {X_val.shape}\n'
)

# Create GPU-compatible DMatrix
dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
dval = xgb.QuantileDMatrix(X_val, label=y_val)

# Train the model
params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train))
}
xgb_pca = xgb.train(params, dtrain, num_boost_round=1000)

# Evaluate the model on validation set
y_pred = xgb_pca.predict(dval)
print(classification_report(y_val, y_pred, target_names=class_names))

end = time.time()
print(f'Time elapsed: {(end-start)/60:.2f} minutes')