#!/usr/bin/env python3

"""
Description: Based on chapter 5
Section 5.1
Introduction of convolution neural networks
Date: 03 - April - 2020
"""

import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# there are 60_000 image of 28x 28 for training
# there are 10_000 image for 28x28 for testing
train_images = train_images.reshape((60_000, 28, 28, 1))
test_images = test_images.reshape((10_000, 28, 28, 1))

# normalize de image from [0 ~ 255] to [0 ~ 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# categorically encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

model = tf.keras.models.Sequential()
# input_shape = (image height, image width, image channel)
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# now the fully connected part
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# for optimizer could be:

#  rms = tf.keras.optimizers.RMSprop(learning_rate=0.01)  # or just optimizer = 'rms' for default values
#  adm = tf.keras.optimizers.Adam(learning_rate=0.001)  # or just optimizer = 'Adam'
#  sgd = tf.keras.optimizers.SGD(learning_rate=0.001)  # or just optimizer = 'SGD'

# for the loss could be:
# mean square error
# categorical crossentropy
# binay crossentropy
# and more

# specify the training configuration

# it is better for now just used the default values so just write de name in ' '
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)

# evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"Accuracy of the model: {test_accuracy}", f"Loss of the model: {test_loss}")
