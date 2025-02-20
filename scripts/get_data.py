import os
import shutil
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