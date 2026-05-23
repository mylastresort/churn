import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray = None,
):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if sample_weights is not None:
        return np.average(per_sample, weights=sample_weights)
    return np.mean(per_sample)


def binary_cross_entropy_derivative(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)


class MyNeuralNet:
    def __init__(
        self,
        hidden_layers_size,
        max_iter=300,
        learning_rate=0.001,
        batch_size=32,
    ):
        self.hidden_layers_size = list(
            hidden_layers_size
            # Convert to list to use later
            # for list concatenation
            # with input and output layer sizes
        )
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = []
        self.biases = []

    def initialize_weights(
        self,
        n_features: int = 1,
    ) -> None:
        layer_sizes = (
            [n_features] + self.hidden_layers_size + [1]
        )  # output layer size is 1 for binary classification

        self.biases = list(range(len(layer_sizes) - 1))
        self.weights = list(range(len(layer_sizes) - 1))

        for i in range(len(layer_sizes) - 1):
            # He initialization to avoid vanishing/exploding gradients
            # create a matrix of weights of layer n and layer n+1
            # for easy matrix multiplication during forward and backward pass
            self.weights[i] = np.random.randn(
                layer_sizes[i], layer_sizes[i + 1]
            ) * np.sqrt(2.0 / layer_sizes[i])
            # create a bias vector for layer n+1
            self.biases[i] = np.zeros((1, layer_sizes[i + 1]))

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        acc = X  # start with input layer
        activated_layers = [acc]  # save activated layers for backpropagation
        weighted_layers = []  # save weighted sums for backpropagation

        for i in range(len(self.weights)):
            # compute weighted sum @ operator is matrix multiplication
            cur = acc @ self.weights[i] + self.biases[i]
            weighted_layers.append(cur)
            cur = relu(cur) if i < len(self.weights) - 1 else sigmoid(cur)
            acc = cur
            activated_layers.append(acc)

        return acc, activated_layers, weighted_layers

    def backward(
        self,
        y: np.ndarray,
        activated_layers: list[np.ndarray],
        weighted_layers: list[np.ndarray],
        sample_weights: np.ndarray = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # size of samples
        m = y.shape[0]

        # apply chain rule for output layer
        # using binary cross-entropy loss for NN binary classification
        dA = binary_cross_entropy_derivative(y, activated_layers[-1])
        # scale each sample's gradient by its class weight before propagating
        if sample_weights is not None:
            dA = dA * sample_weights
        dZ = dA * sigmoid_derivative(weighted_layers[-1])

        # compute gradients - we transpoose for valid matrix multiplication
        # (batch_size, n_weights).T @ (batch_size, n_output_weights=1) -> (n_weights, n_output_weights=1)
        # we divide by m to get average gradient over all samples
        # we apply this chain rule for all layers to update weights
        dw = activated_layers[-2].T @ dZ / m
        # dZ is (batch_size, n_output_weights=64) we sum over y axis to get (n_output_weights=64,) - axis=0 means sum columns
        db = np.sum(dZ, axis=0) / m

        # print(
        #     dZ.shape,
        #     activated_layers[-2].T.shape,
        #     dw.shape,
        #     db.shape,
        #     flush=True,
        # )
        # (1024, 1) (64, 1024) (64, 1) (1,)

        grads_w = [dw]
        grads_b = [db]

        # iterate backwards through hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            # we compute activation derivative of current layer backwards
            dA = dZ @ self.weights[i].T
            # we apply chain rule and multiply activation derivative with derivative of current layer
            dZ = dA * relu_derivative(weighted_layers[i - 1])

            # compute gradients
            dw = activated_layers[i - 1].T @ dZ / m
            db = np.sum(dZ, axis=0) / m

            # print(
            #     dZ.shape,
            #     activated_layers[i - 1].T.shape,
            #     dw.shape,
            #     db.shape,
            #     flush=True,
            # )
            # (1024, 64) (64, 1024) (64, 64) (64,)
            # (1024, 64) (32, 1024) (32, 64) (64,)

            # append gradients
            grads_w.append(dw)
            grads_b.append(db)

        grads_w.reverse()
        grads_b.reverse()
        return grads_w, grads_b

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_weight: dict = None,
        validation_data: tuple[pd.DataFrame, pd.Series] = None,
    ) -> None:
        # convert pandas dataframe to ndarrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        # reshape to column vector to compare with output layer
        y_array = (
            y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        )

        # build per-sample weight array from class_weight dict (fit on train only)
        if class_weight is not None:
            sw_array = np.array(
                [class_weight[int(yi)] for yi in y_array.flatten()]
            ).reshape(-1, 1)
        else:
            sw_array = None

        # initialize weights using layer sizes
        self.initialize_weights(n_features=X_array.shape[1])

        n_samples = X_array.shape[0]  # number of input rows
        for epoch in range(self.max_iter):
            # shuffle data in each epoch
            indices = np.random.permutation(n_samples)
            # save shuffled data views for training
            X_shuffled = X_array[indices]
            y_shuffled = y_array[indices]
            sw_shuffled = sw_array[indices] if sw_array is not None else None

            epoch_loss = 0  # to average loss after each batch training
            n_batches = 0

            # Mini-batch training
            # updates the weights in each batch instead of one epoch
            for start_idx in range(0, n_samples, self.batch_size):
                # compute the end index of the batch
                end_idx = min(start_idx + self.batch_size, n_samples)
                # get the batch data views
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                sw_batch = (
                    sw_shuffled[start_idx:end_idx] if sw_shuffled is not None else None
                )

                # forward propagation
                y_pred, activated_layers, weighted_layers = self.forward(X_batch)

                # get the loss
                batch_loss = binary_cross_entropy(y_batch, y_pred, sw_batch)
                epoch_loss += batch_loss

                n_batches += 1

                # backward propagation to get gradients
                grads_weights, grads_biases = self.backward(
                    y_batch, activated_layers, weighted_layers, sw_batch
                )

                # update weights using gradient descent
                # gives same result as Adam optimizer
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grads_weights[i]
                    self.biases[i] -= self.learning_rate * grads_biases[i]

            avg_epoch_loss = epoch_loss / n_batches
            acc = self.score(X, y)
            auc = self.auc_score(X, y)
            val_acc = self.score(*validation_data) if validation_data else 0
            val_auc = self.auc_score(*validation_data) if validation_data else 0
            print(
                f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_epoch_loss:.4f} - Acc: {acc:.4f} - AUC: {auc:.4f} - Val Acc: {val_acc:.4f} - Val AUC: {val_auc:.4f}",
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_proba, _, _ = self.forward(X_array)
        return y_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        # convert pandas dataframe to ndarrays
        y_array = (
            y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        )
        y_pred = self.predict(X)
        # compute accuracy
        accuracy = np.mean(y_pred == y_array)
        return accuracy

    def auc_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_array = (
            y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        )
        return roc_auc_score(y_array, self.predict_proba(X))
