import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report, precision_recall_fscore_support

(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)
print(test_x.shape)

# 1. Data preprocessing:
# 1.1. Data Normalization:
train_x = train_x/255
test_x = test_x/255

# 1.2. Data flattening:
# displaying before flattening:
train_x_flattened = train_x.reshape(len(train_x), 28*28)
test_x_flattened = test_x.reshape(len(test_x), 28*28)

# displaying after flattening:
print(train_x_flattened.shape)
print(test_x_flattened.shape)

# 2. ANN Model:
# 2.1. Building the Model:
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),  # one hidden layer with 100 neurons
    keras.layers.Dense(10, activation='sigmoid')  # output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x_flattened, train_y, epochs=10, validation_split=0.3)

# 2.1. Line the Accuracy vs Epochs:
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="Test accuracy")
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()

# 3. Model Prediction:
prediction = model.predict(test_x_flattened)

# 3.1. prediction and plotting a single image:
plt.matshow(test_x[0])  # displaying the first image of the test dataset
plt.show()
print(test_y[0])  # displaying the label of the first image of the test dataset
print(prediction[0])  # displaying the prediction on an 10 element-array (as # of output neurons)
print(f'Prediction is {np.argmax(prediction[0])}')  # printing the maximum value of the 10 element-array

# 3.2. prediction and plotting a bunch of images:
predicted_values = []
for i in range(9):
    predicted_values.append(np.argmax(prediction[i]))
    plt.subplot(330 + i + 1)
    plt.imshow(test_x[i])
    plt.title(predicted_values[i])
print(predicted_values)
plt.tight_layout()
plt.show()

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
test_loss, test_acc = model.evaluate(test_x_flattened, test_y)
print(f'Model ACCURACY is {test_acc}')
print(f'Model LOSS is {test_loss}')
print(classification_report(test_y, complete_predicted_values))

# 4.3. TPR (Sensitivity) & FPR (Specificity): with many thresholds:
result = []
for i in range(10):
    prec_ann, recall_ann, _, _ = precision_recall_fscore_support(np.array(test_y) == i,
                                                        np.array(complete_predicted_values) == i,
                                                        pos_label=True, average=None)
    result.append([i, recall_ann[0], recall_ann[1]])

print(pd.DataFrame(result, columns=['class', 'sensitivity', 'specificity']))