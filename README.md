# landscapes
### Image headline classification using residual networks  
This project aims to classify landscape images into one of six categories. The data consist of over 14,000 labelled images, initially gather by Intel as part of a competition with Analytics Vidhya: Datahack [1]. The primary model under consideration is ResNet50 [2].

Below is a description of the various scripts (all saved under the scripts folder). Configuration is handled via the `scripts/config.json` file. The overview.ipynb provides an end-to-end view of the development procedure, covering exploratory data analysis, benchmark and baseline training, tuning for performance, and final evaluation.

`get_data.py`  
This script downloads the images from Kaggle, then splits it them into training, validation and testing subsets, all captured under distinct directories. Note that the kaggle data includes an additional 3,000 images with*out* labels, intended for the initial competition. Because the competition has concluded and these images have no labels, they are discarded. Data is saved in a newly created `data` directory.  
Note that this script uses the Kaggle API for Python, and requires that kaggle.json be saved to your PATH for authentication. See instructions here: https://python.plainenglish.io/how-to-use-the-kaggle-api-in-python-4d4c812c39c7.

`preprocess.py`  
This script preprocesses data for both ResNet50 and a benchmark model. For ResNet50, the script leverages PyTorch dataloaders and specifies a series of transformations to prepare the images for ResNet50. The image sizes are standardized, arrays converted to tensors, and values across each RGB channel normalized. Additional randomized transformations are specified for the training subset. For the benchmark model, a flat, tabular dataset is generated.  
Key processing parameters — namely `num_workers`, `batch_size`, and the random transformations — can be adjusted in `config.json`.

`train_benchmark.py`  
This script trains a benchmark model using XGBoost. Because of the size of the flattened images (over 150,000 columns), PCA is performed to reduce dimensionality. The reduced feature set is then used to train an XGB classifier, and evaluation metrics are provided.

`train_resnet.py`  
This scripts trains the ResNet50 model. The `config.json` file allows for configuration of the number of epochs, the learning rate, the dropout rate (for regularization), and the optimizer. The final model will be saved in the `models` directory.

`error_analysis.py`  
This script provides details on errors in the most recently trained models. It generated classification reports, as well as plots of erroneous predictions across each class.

`tuning.py`  
This script iterates through various configurations of hyperparameters and logs the models' resulting performance for comparison, under the `logs` directory. The script leverages the Optuna library.

Sources  
[1] Intel and Analytics Vidhya: Datahack. "Intel Image Classification Challenge." Sourced from https://www.kaggle.com/datasets/puneet6060/intel-image-classification.  
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". December 2015. https://arxiv.org/abs/1512.03385.  
