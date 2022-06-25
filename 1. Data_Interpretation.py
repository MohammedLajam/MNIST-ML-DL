from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np

# 1. displaying the shape of the dataset:
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(len(train_x))  # displaying the number of the data in train_x
print(train_x[0].shape)  # displaying the pixels of the first image
print(train_x[0])  # representing the first image in a 2 array
print(f'Train_x: {train_x.shape}')
print(f'Train_y: {train_y.shape}')
print(f'Test_x: {test_x.shape}')
print(f'Test_y: {test_y.shape}')

# 2. plotting the dataset:
# 2.1. plotting a single image:
plt.matshow(train_x[0])
plt.show()
print(train_y[0])

# 2.2. plotting several images in a single window:
for i in range(9):
    plt.subplot(330 + i + 1)
    plt.imshow(train_x[i])
plt.show()
print(train_y[:9])

# 3. Data Visualization:
total_classes = 10

# 3.1. Shape of training data
total_examples, img_length, img_width = train_x.shape
# Print the statistics
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# 3.2. Scatter Plots in matplotlib and Seaborn (PCA):
# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(train_x, (train_x.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Plotting the Scatter Plot:
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=train_y, s=5)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()

# 3.3. 3D Scatter Plot:
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_y, s=1)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.show()

# 4. Histogram:
