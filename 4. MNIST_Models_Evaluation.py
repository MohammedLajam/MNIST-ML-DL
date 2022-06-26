import keras
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dropout, Activation, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# loading the training and test sets:
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Model 1: ANN:
# 1. Data preprocessing:
# 1.1. Data Normalization:
train_x = train_x/255
test_x = test_x/255

# 1.2. Data flattening:
train_x_flattened = train_x.reshape(len(train_x), 28*28)
test_x_flattened = test_x.reshape(len(test_x), 28*28)

# 2. ANN Model:
# 2.1. Building the Model:
model1 = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),  # one hidden layer with 100 neurons
    keras.layers.Dense(10, activation='sigmoid')  # output layer
])

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model1.fit(train_x_flattened, train_y, epochs=10, validation_split=0.3)

# 3. Model Prediction:
prediction = model1.predict(test_x_flattened)

# 4. Model Evaluation:
# 4.1. Confusion Matrix:
# converting all the predicted values in the test dataset into one value in array
complete_predicted_values = []
for i in range(10000):
    complete_predicted_values.append(np.argmax(prediction[i]))
print(complete_predicted_values)  # printing the complete predicted value in a list

cm = tf.math.confusion_matrix(labels=test_y, predictions=complete_predicted_values)
print(cm)  # printing the confusion matrix in the terminal

# plotting the confusion matrix using seaborn and matplotlib
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 4.2. Accuracy, Precision, Recall and F1 score:
# For all the classes accumulated:
test_loss, test_acc = model1.evaluate(test_x_flattened, test_y)
print(f'Model ACCURACY is {test_acc}')
print(f'Model LOSS is {test_loss}')
print(classification_report(test_y, complete_predicted_values))

# 4.3. TPR (Sensitivity) & FPR (Specificity): with many thresholds
result = []
for i in range(10):
    prec, recall, _, _ = precision_recall_fscore_support(np.array(test_y) == i,
                                                        np.array(complete_predicted_values) == i,
                                                        pos_label=True, average=None)
    result.append([i, recall[0], recall[1]])

print(pd.DataFrame(result, columns=['class', 'sensitivity', 'specificity']))

# 4.4. ROC/AUC to evaluate different models:
