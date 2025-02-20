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

# # Perform PCA to generate features
# pca_datasets = {}
# pca = PCA(n_components=1000)

# for subset in ['train', 'val', 'test']:
#     pca.fit(tabular_datasets[subset][0])
#     pca_datasets.update({subset: (
#         pca.transform(tabular_datasets[subset][0]), 
#         tabular_datasets[subset][1]
#     )})

# print(
#     f'Train PCA shape: {pca_datasets["train"][0].shape}\n'
#     f'Val PCA shape: {pca_datasets["val"][0].shape}\n'
#     f'Test PCA shape: {pca_datasets["test"][0].shape}'
# )

# # Set up pca data
# X_train, y_train = pca_datasets['train']
# X_val, y_val = pca_datasets['val']

X_train, y_train = tabular_datasets['train']
X_val, y_val = tabular_datasets['val']

print(
    f'X Train shape: {X_train.shape}\n'
    f'X Val shape: {X_val.shape}\n'
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