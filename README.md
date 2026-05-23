# Churn Prediction

Binary classification project predicting bank client churn using neural networks.

Check [notebooks/churn.ipynb](notebooks/churn.ipynb) for models accuracy and additional performed tasks.

## Structure

| Path                                           | Description                                     |
| ---------------------------------------------- | ----------------------------------------------- |
| [data/](data/)                                 | Train / test CSVs (bank client features)        |
| [src/features.py](src/features.py)             | Preprocessing & feature engineering             |
| [src/my_neural_net.py](src/my_neural_net.py)   | NumPy neural net built from scratch             |
| [src/NN_GridSearch.py](src/NN_GridSearch.py)   | Keras residual net + hyperparameter grid search |
| [src/DataSplit.py](src/DataSplit.py)           | Train/val/test split container                  |
| [src/utils.py](src/utils.py)                   | Shared helpers                                  |
| [models/](models/)                             | Saved Keras models                              |
| [predictions/](predictions/)                   | Output CSVs                                     |
| [notebooks/churn.ipynb](notebooks/churn.ipynb) | Exploratory analysis & experiments              |

## Stack

TensorFlow/Keras · scikit-learn · pandas · NumPy

## Quickstart

```bash
uv sync
jupyter lab notebooks/churn.ipynb
```
