import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout

# loading the dataset:
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# reshaping the training and test dataset:
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

# Normalization:
train_x = train_x.astype('float32')/255
test_x = test_x.astype('float32')/255

# plotting sample of the dataset:
fig = plt.figure(figsize=(5, 3))
for i in range(15):
    ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(train_x[i]), cmap='gray')
    ax.set_title(train_y[i])
plt.show()

# shape of the dataset:
print(train_x.shape[1:])

# Building a Model:
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_x.shape[1:]))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(48, (2, 2), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# train the model:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, batch_size=128, verbose=2, validation_split=0.1)

# Model Evaluation:
print("Model Evaluation")
test_loss, test_acc = model.evaluate(test_x, test_y)
print(f'Test loss on 10000 test samples is {test_loss}')
print(f'Validation accuracy on 10000 test samples is {test_acc*100}')

# Predictions:
prediction = model.predict(test_x)
plt.imshow(test_x[0])  # displaying the first image of the test dataset
plt.show()
print(test_y[0])
print(prediction[0])
print(f'Prediction is {np.argmax(prediction[0])}')
