#!/usr/bin/env python3

"""
Description: Based on chapter 7
Section 7.1
Date: 11 - April - 2020

Basically here we show the use of callback in Keras, which can be used as:
* Model Checkpointing
* Early stopping
* To adjust parameters if the learning process does not improve

This is particularly useful to fix some problems at an early state

Also I include here TensorBoard
"""

import tensorflow as tf

# -------------------------------data----------------------------
train_dir = '/home/david/PycharmProjects/datasets/pet_images_data/training_dataset'
validation_dir = '/home/david/PycharmProjects/datasets/pet_images_data/validation_dataset'

#  Before start using TensorBoar, we need to create a directory where we will store the log files it generates
#  I created the directory david_TensorBoard_dir

#  ----------------- Create de model ----------------------------------------------

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

rms = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])

#  --------------- Data preprocessing  --------------------------

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                     target_size=(150, 150),
                                                     batch_size=20,
                                                     class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                     target_size=(150, 150),
                                                     batch_size=20,
                                                     class_mode='binary')


# ------------------------- Call back ------------------------------
# ------------------------- model checkpoint// early stopping // TensorBoard

# this interrupts training when the accuracy has stopped improving for more than one epoch// patient = wait epochs
# min_delta = Minimum change in the monitored quantity to qualify as an improvement,
# i.e. an absolute change of less than min_delta, will count as no improvement.
# monitor + also could be 'val_loss'

# the second line, saves the weights after every epochs, this wont overwrite the model file unless val_loss has improved
callback_list = [
                tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.05, patience=1),
                tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True),
                tf.keras.callbacks.TensorBoard(log_dir='david_TensorBoard_dir', histogram_freq=1, embeddings_freq=1)
                ]

# ------------------------ train the model ---------------------------------

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              callbacks=callback_list,
                              validation_data=validation_generator,
                              validation_steps=50)
