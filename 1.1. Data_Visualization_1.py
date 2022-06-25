# https://machinelearningmastery.com/data-visualization-in-python-with-matplotlib-seaborn-and-bokeh/
# Data Visualization in Python with matplotlib, Seaborn, and Bokeh
# Plotting the first 3 classes:

# Importing from tensorflow and keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras import utils
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
# For math operations
import numpy as np
# For plotting with matplotlib
import matplotlib.pyplot as plt
# For plotting with seaborn
import seaborn as sns
# For plotting with bokeh
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem
# For pandas dataframe
import pandas as pd

# load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Shape of training data
total_examples, img_length, img_width = x_train.shape
# Print the statistics
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Figures in matplotlib: plotting bunch of images:
img_per_row = 8
fig, ax = plt.subplots(nrows=2, ncols=img_per_row, figsize=(18, 4), subplot_kw=dict(xticks=[], yticks=[]))
for row in [0, 1]:
    for col in range(img_per_row):
        ax[row, col].imshow(x_train[row*img_per_row + col].astype('int'))
plt.show()

# Scatter Plots in matplotlib and Seaborn (PCA):
# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Plotting the Scatter Plot:
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()

# 3D Scatter Plot:
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_labels, s=1)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.show()

