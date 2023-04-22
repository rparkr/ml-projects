# ML from Scratch
This folder contains notebooks where I implement various machine learning algorithms from scratch using only `NumPy`. Each notebook contains detailed documentation, mathematical explanations, and demos to experiment with the algorithms.

I built these projects to practice the skill of taking mathematical formulas and turning them into vectorized code. As such, I did not review anyone else's code related to these algorithms but rather studied their mathematical formulations.

Currently, the algorithms I have implemented are:
- [Neural network](./neural_network_from_scratch.ipynb)
- [Linear regression](./linear_regression_from_scratch.ipynb)
- [Logistic regression](./logistic_regression_from_scratch.ipynb)

## Neural network from scratch
A configurable fully-connected neural network that can be used for regression or classification tasks. This is an all-purpose algorithm that can be applied to many tasks like word embeddings, time series predictions, or multiclass classification (especially when using tabular data).

**Key algorithms used include**: gradient descent and backpropagation, cross entropy loss, L2 regularization, sigmoid and ReLU activations, forward propagation, normalization, and a training and validation loop.

Results on sample data:
|   Classification   |   Regression   |
|:------------------:|:--------------:|
| ![](/images/nn_classification_test3.png) | ![](/images/nn_regression.png) |

## Logistic regression from scratch
A logistic regression model for binary classification using gradient descent to optimize the model's parameters. Uses any number of input features.

Training progress on a sample dataset:  
![](../images/logistic_regression_training.gif)

## Linear regression from scratch
A linear regression model trained using gradient descent with optional early stopping. Uses any number of input features (i.e., multilinear regression).

**Key algorithms used include**: mean squared error loss, model training through gradient descent.

Results from a multilinear regression demo: notice that `y_hat` (color gradient) closely matches `y` (vertical axis).
![](/images/linear_regression.png)
