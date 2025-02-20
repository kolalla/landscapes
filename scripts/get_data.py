import os
import shutil
import random
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

# authentication
api = KaggleApi()
api.authenticate()

# clear out data directory
if not os.path.exists('data'):
    os.makedirs('data')
else:
    shutil.rmtree(f'data')
    os.makedirs('data')

# download dataset with Kaggle API
api.dataset_download_files('puneet6060/intel-image-classification')

# designate downloaded file as zip, and unzip
zf = ZipFile('intel-image-classification.zip')
zf.extractall('data')
zf.close()

# delete downloaded zip and extracted csv - keep your directory clean!
os.remove('intel-image-classification.zip')

# clean up folder structure
for subset in ['train', 'test']:

    if not os.path.exists(f'data/{subset}'):
        os.makedirs(f'data/{subset}')

    base_path = f'data/seg_{subset}'
    source_path = f'data/seg_{subset}/seg_{subset}'
    target_path = f'data/{subset}'

    num_files = 0
    for folder in os.listdir(source_path):
        shutil.move(os.path.join(source_path, folder), os.path.join(target_path, folder))
        num_files += len(os.listdir(os.path.join(target_path, folder)))
    print(f'Number of {subset} files: {num_files}')

    os.rmdir(source_path)
    os.rmdir(base_path)

num_pred_files = len(os.listdir('data/seg_pred/seg_pred'))
print(f'Number of pred files: {num_pred_files}')
for img in os.listdir('data/seg_pred/seg_pred'):
    os.remove(f'data/seg_pred/seg_pred/{img}')
os.rmdir('data/seg_pred/seg_pred')
os.rmdir('data/seg_pred')

# create validation set
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