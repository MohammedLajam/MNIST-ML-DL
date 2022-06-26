import keras
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dropout, Activation, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(len(train_x), 28*28)
test_x = test_x.reshape(len(test_x), 28*28)


