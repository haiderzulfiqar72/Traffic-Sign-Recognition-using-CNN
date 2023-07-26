import pickle
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm
from numpy import loadtxt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from keras.utils import np_utils
import keras
from sklearn.model_selection import train_test_split
from plot_keras_history import show_history, plot_history
from keras.layers import Conv2D, Flatten, Dropout
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from skimage.io import imread_collection

img_path = "E:/Study Material/Tampere - Grad/Studies/Year I/Period IV/Pattern Recognition and Machine Learning - DATA.ML.200/Excercises/Excercise Week 4/GTSRB_subset_2/GTSRB_subset_2/"

classes = 2
class_labels = ['class1', 'class2']

# Data and Labels
img_list = []
label_list = []
for label in class_labels:
    label_path = os.path.join(img_path, label)
    for img_file in os.listdir(label_path):
        img = plt.imread(os.path.join(label_path, img_file))
        img_list.append(img)
        label_list.append(label)
data = np.array(img_list)
labels = np.array(label_list)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, label_list, test_size=0.2)

# One Hot Encoding on the training and testing labels separately
class_dict = {'class1': 0, 'class2': 1}
y_train_encoded = [class_dict[label] for label in y_train]
y_test_encoded = [class_dict[label] for label in y_test]
y_train_one_hot = keras.utils.to_categorical(np.array(y_train_encoded), num_classes=2)
y_test_one_hot = keras.utils.to_categorical(np.array(y_test_encoded), num_classes=2)

# Simple Sequential structure)
model = tf.keras.models.Sequential()

# Add a convolutional layer with 10 filters of size (3, 3), stride 2, and ReLU activation function. 
# The input shape is (64, 64, 3)
model.add(tf.keras.layers.Conv2D(10, (3, 3), strides=(2, 2), activation='relu', input_shape=(64, 64, 3)))
print(model.output_shape)

# Add batch normalization layer to improve training
model.add(BatchNormalization())

# Add a max pooling layer with pool size (2, 2)
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
print(model.output_shape)

# Add another convolutional layer with the same parameters as the first
model.add(tf.keras.layers.Conv2D(10, (3, 3), strides=(2, 2), activation='relu'))
print(model.output_shape)

# Add another max pooling layer with pool size (2, 2)
# Flatten the output of the max pooling layer
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())

# Add a dense layer with 2 neurons and sigmoid activation function
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

print(model.summary())

# Compile the model
model.compile(optimizer='SGD', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
history = model.fit(x_train/255, y_train_one_hot, batch_size=32, epochs=20)

show_history(history)
plot_history(history, path="standard.png")
plt.close()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test/255, y_test_one_hot, verbose=2)
print('\nTest accuracy:', test_acc)

# plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# plotting graphs for loss 
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plot_history(history, path="standard.png")
plt.close()
