# COMP551 - Modified MNIST Challenge

## Prerequisites

Dependences: 

```
Python 3.4+
h5py (pip)
opencv-python (pip)
TensorFlow 1.4+
```

To train the CNN, the following files should be present:

```
data/train_x.csv
data/train_y.csv
data/test_x.csv
```

These files can be retrieved from the COMP 551 [Kaggle competiton](https://www.kaggle.com/c/comp551-modified-mnist/data).

## Train the model

To train the model, simply run the following script: `python model_baseline.py`. This will save a model file to `data/temp_model.hdf5`.

To monitor training progress, run: `tensorboard --logdir=logs`

## Classify the data

To classify the test data, run: `python classify_data.py [optional_model_file.hdf5 (defaults to data/temp_model.hdf5)] [optional_output_file (defaults to data/test_y.csv)]`.
