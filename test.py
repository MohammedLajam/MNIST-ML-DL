import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dropout, Activation, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# 1. Data Visualization:
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 2. Data Preprocessing:
# 2.1. Normalization:
train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)

# 2.2. Resizing the images to make it suitable for convolutional operation:
IMG_SIZE = 28
train_x_r = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x_r = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# 3. Building a CNN Model:
model = keras.models.Sequential()

# First Convolutional Layer:
model.add(Conv2D(32, (3, 3), input_shape=train_x_r.shape[1:]))
model.add(Activation("relu"))  # Activation function to make non-linear
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer:
model.add(Conv2D(48, (3, 3), input_shape=train_x_r.shape[1:]))
model.add(Activation("relu"))  # Activation function to make non-linear
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layer (Flatten)
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# Output layer:
model.add(Dense(10))
model.add(Activation("Softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
history = model.fit(train_x_r, train_y, epochs=2, validation_split=0.3)

# 4. Model prediction:
prediction = model.predict(test_x_r)

# 5. Model Evaluation:
# 5.1. Accuracy, Precision, Recall and F1 score:
# converting all the predicted values in the test dataset into one value in array
complete_predicted_values = []
for i in range(10000):
    complete_predicted_values.append(np.argmax(prediction[i]))

print("Model Evaluation")
test_loss, test_acc = model.evaluate(test_x_r, test_y)
print(f'Test loss on 10000 test samples is {test_loss}')
print(f'Validation accuracy on 10000 test samples is {test_acc}')
print(classification_report(test_y, complete_predicted_values))

# 5.2. Confusion Matrix
cm = tf.math.confusion_matrix(labels=test_y, predictions=complete_predicted_values)
print(cm)  # printing the confusion matrix in the terminal

# plotting the confusion matrix using seaborn and matplotlib
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 5.3. TPR (Sensitivity) & FPR (Specificity): with many thresholds
result = []
for i in range(10):
    prec, recall, _, _ = precision_recall_fscore_support(np.array(test_y) == i,
                                                        np.array(complete_predicted_values) == i,
                                                        pos_label=True, average=None)
    result.append([i, recall[0], recall[1]])

print(pd.DataFrame(result, columns=['class', 'sensitivity', 'specificity']))

# 5.4. ROC/AUC to evaluate different models:

