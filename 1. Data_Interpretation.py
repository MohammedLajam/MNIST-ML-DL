from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

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