# Models Evaluation of ANN and CNN for MNIST Dataset (Hand-Written-Digits):

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# loading the training and test sets:
(train_x_ann, train_y_ann), (test_x_ann, test_y_ann) = mnist.load_data()
(train_x_cnn, train_y_cnn), (test_x_cnn, test_y_cnn) = mnist.load_data()

# 1. Data Normalization:
# 1.1. ANN:
train_x_ann = train_x_ann/255
test_x_ann = test_x_ann/255
train_x_ann = train_x_ann.reshape(len(train_x_ann), 28*28)
test_x_ann = test_x_ann.reshape(len(test_x_ann), 28*28)

# 1.2. CNN:
train_x_cnn = tf.keras.utils.normalize(train_x_cnn, axis=1)
test_x_cnn = tf.keras.utils.normalize(test_x_cnn, axis=1)
IMG_SIZE = 28
train_x_cnn = np.array(train_x_cnn).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x_cnn = np.array(test_x_cnn).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# 2. Building a Model:
# 2.1. ANN:
model_ann = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),  # one hidden layer with 100 neurons
    keras.layers.Dense(10, activation='sigmoid')  # output layer
])
model_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_ann = model_ann.fit(train_x_ann, train_y_ann, epochs=102, validation_split=0.3)

# 2.2. CNN:
model_cnn = keras.models.Sequential()
model_cnn.add(Conv2D(32, (3, 3), input_shape=train_x_cnn.shape[1:]))
model_cnn.add(Activation("relu"))  # Activation function to make non-linear
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Conv2D(48, (3, 3), input_shape=train_x_cnn.shape[1:]))
model_cnn.add(Activation("relu"))  # Activation function to make non-linear
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Flatten())
model_cnn.add(Dense(500))
model_cnn.add(Activation("relu"))
model_cnn.add(Dense(10))
model_cnn.add(Activation("Softmax"))

model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
history_cnn = model_cnn.fit(train_x_cnn, train_y_cnn, epochs=5, validation_split=0.3)

# 3. Models Predictions:
# 3.1. ANN:
predictions_ann = model_ann.predict(test_x_ann)
complete_predicted_values_ann = []
for i in range(10000):
    complete_predicted_values_ann.append(np.argmax(predictions_ann[i]))

# 3.2. CNN:
predictions_cnn = model_cnn.predict(test_x_cnn)
complete_predicted_values_cnn = []
for i in range(10000):
    complete_predicted_values_cnn.append(np.argmax(predictions_cnn[i]))

# 4. Models Evaluations:
print("Models Evaluation")
# 4.1. Confusion Matrix:
cm_ann = tf.math.confusion_matrix(labels=test_y_ann, predictions=complete_predicted_values_ann)
cm_cnn = tf.math.confusion_matrix(labels=test_y_cnn, predictions=complete_predicted_values_cnn)

# plotting the confusion matrix using seaborn and matplotlib (ANN)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_ann, annot=True, fmt='d')
plt.title('Confusion Matrix (ANN)')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# plotting the confusion matrix using seaborn and matplotlib (CNN)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_cnn, annot=True, fmt='d')
plt.title('Confusion Matrix (CNN)')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 4.2. Accuracy, Precision, Recall and F1 score
test_loss_ann, test_acc_ann = model_ann.evaluate(test_x_ann, test_y_ann)
print(f'ANN ACCURACY is {test_acc_ann}')
print(f'ANN LOSS is {test_loss_ann}')
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(test_x_cnn, test_y_cnn)
print(f'CNN Accuracy is {test_acc_cnn}')
print(f'CNN LOSS is {test_loss_cnn}')
print("Classification report (ANN):")
print(classification_report(test_y_ann, complete_predicted_values_ann))
print("Classification report (CNN):")
print(classification_report(test_y_cnn, complete_predicted_values_cnn))

# 4.3. TPR (Sensitivity) & FPR (Specificity): with many thresholds
# ANN:
ss_ann = []
for i in range(10):
    prec_ann, recall_ann, _, _ = precision_recall_fscore_support(np.array(test_y_ann) == i,
                                                        np.array(complete_predicted_values_ann) == i,
                                                        pos_label=True, average=None)
    ss_ann.append([i, recall_ann[0], recall_ann[1]])

print('Sensitivity and Specificity (ANN)')
print(pd.DataFrame(ss_ann, columns=['class', 'sensitivity', 'specificity']))

# CNN:
ss_cnn = []
for i in range(10):
    prec_cnn, recall_cnn, _, _ = precision_recall_fscore_support(np.array(test_y_cnn) == i,
                                                        np.array(complete_predicted_values_cnn) == i,
                                                        pos_label=True, average=None)
    ss_cnn.append([i, recall_cnn[0], recall_cnn[1]])

print('Sensitivity and Specificity (CNN)')
print(pd.DataFrame(ss_cnn, columns=['class', 'sensitivity', 'specificity']))