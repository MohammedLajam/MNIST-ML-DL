import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dropout, Activation, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report

# 1. Data Visualization:
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)
# 1.1. Visualizing one image and its label.
plt.imshow(train_x[0])  # plotting the original image
plt.imshow(train_x[0], cmap=plt.cm.binary)  # plotting the image in binary colors
plt.show()
print(train_y[0])

# 2. Data Preprocessing:
# Checking the values of the pixels before Normalization:
print(train_x[0])

# 2.1. Normalization:
train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)
plt.imshow(train_x[0], cmap=plt.cm.binary)
print(train_x[0])

# 2.2. Resizing the images to make it suitable for convolutional operation:
IMG_SIZE = 28
train_x_r = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x_r = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(f'Training sample dimensions: {train_x_r.shape}')
print(f'Test sample dimensions: {test_x_r.shape}')

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

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
history = model.fit(train_x_r, train_y, epochs=10, validation_split=0.3)

# 2.1. Line the Accuracy vs Epochs:
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="Test accuracy")
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()

# 4. Model prediction:
print('Model Predictions')
prediction = model.predict(test_x_r)
# 4.1. prediction and plotting a single image:
plt.imshow(test_x_r[0])  # displaying the first image of the test dataset
plt.show()
print(test_y[0])
print(prediction[0])
print(f'Prediction is {np.argmax(prediction[0])}')

# 4.2. prediction and plotting a bunch of images:
predicted_values = []
for i in range(9):
    predicted_values.append(np.argmax(prediction[i]))
    plt.subplot(330 + i + 1)
    plt.imshow(test_x[i])
print(predicted_values)
plt.show()

# 5. Model Evaluation:
# 5.1. Accuracy, Precision, Recall and F1 score:
# converting all the predicted values in the test dataset into one value in array
complete_predicted_values = []
for i in range(10000):
    complete_predicted_values.append(np.argmax(prediction[i]))
print(complete_predicted_values)  # printing the complete predicted value in a list

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

# 5.4. ROC/AUC to evaluate different models:

