# Machine Learning projects
This repository contains an assortment of my projects in machine learning. Here's a brief overview of some of what you'll find here:

# ML from Scratch
The [ml_from_scratch](/ml_from_scratch/) folder contains notebooks where I implement various machine learning algorithms from scratch using only `NumPy`. Each notebook contains detailed documentation, mathematical explanations, and demos to experiment with the algorithms.

I built these projects to practice the skill of taking mathematical formulas and turning them into vectorized code. As such, I did not review anyone else's code related to these algorithms but rather studied their mathematical formulations.

Currently, the algorithms I have implemented are:
- [Neural network](/ml_from_scratch/neural_network_from_scratch.ipynb)
- [Linear regression](/ml_from_scratch/linear_regression_from_scratch.ipynb)
- [Logistic regression](/ml_from_scratch/logistic_regression_from_scratch.ipynb)

## Neural network from scratch
A configurable fully-connected neural network that can be used for regression or classification tasks. This is an all-purpose algorithm that can be applied to many tasks like word embeddings, time series predictions, or multiclass classification (especially when using tabular data).

**Key algorithms used include**: gradient descent and backpropagation, cross entropy loss, L2 regularization, sigmoid and ReLU activations, forward propagation, normalization, and a training and validation loop.

Results on sample data:
|   Classification   |   Regression   |
|:------------------:|:--------------:|
| ![](/images/nn_classification_test3.png) | ![](/images/nn_regression.png) |

## Logistic regression from scratch
A logistic regression model for binary classification using gradient descent to optimize the model's parameters. Uses any number of input features.

Training progress on a sample dataset, showing how the decision boundary is updated through gradient descent:  
![](/images/logistic_regression_training.gif)

## Linear regression from scratch
A linear regression model trained using gradient descent with optional early stopping. Uses any number of input features (i.e., multilinear regression).

**Key algorithms used include**: mean squared error loss, model training through gradient descent.

Results from a multilinear regression demo: notice that `y_hat` (color gradient) closely matches `y` (vertical axis).
![](/images/linear_regression.png)

# Dominant Color Extraction
In my [dominant color extraction](/dominant_color_extraction.ipynb) notebook, I use the K-Means algorithm to estimate the main color for an image, which I then apply to the task of detecting a vehicle's color given an image of that vehicle. I used this technique in my [vehicle specs](https://github.com/rparkr/ML-practice/blob/main/Vehicle%20specs/Final%20project/ML_pipeline_Vehicle_Specs.ipynb) project as part of a feature engineering pipeline to augment a dataset with vehicle color information.

![](/images/dominant_color_extraction.png)


# Extractive text summarizing
In [this notebook](/extractive_text_summarizing.ipynb), adapted from a tutorial by Usman Malik, I implement a method for extractive text summarizing and use it to summarize Wikipedia articles.


# Craiyon Text2Image
I created an [sort-of API to generate images](/craiyon_text2image.ipynb) using the Craiyon text-to-image model, available at: https://www.craiyon.com/

![](/images/craiyon_text2image_demo.png)

# Text analytics assignments
I took a computational linguistics course and compiled my work on course assignments into a single notebook. In [this notebook](/text_analytics_assignments.ipynb), I compute precision and recall for varioius NLP algorithms, compare categories and word frequency distributions across a corpus, classify sentiment, use part-of-speech tagging, predict categories using synsets from WordNet, and use LDA to visualize topics, compute the number of topcis using perplexity and coherence, and use a lot of RegEx, web scraping, NLTK, and topic modeling with Gensim.

![](/images/text_analytics_perplexity_and_coherence.png "Perplexty and coherence chart for topic modeling")

![](/images/text_analytics_lda.png "Topic modeling with LDA")

![](/images/text_analytics_feature_correlation.png "Correlation between features and label for a reviews dataset")