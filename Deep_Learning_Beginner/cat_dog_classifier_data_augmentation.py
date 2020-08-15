#!usr/bin/env python3

"""
Date: 07 - April - 2020

Description: Based on chapter 5, using convnets with small datasets section 5.2
Aqui  se toman las medidas necesarias para corregir el overfitting espcificamente se usa data augmentation
Esto es muy utilizado en computer vison.
Este codigo es una continuacion del codigo planteado en cat_dog_classifier.py

Data augmentation takes the approach of generating more training data from existing training samples, by
augmenting the samples via a number of random transformations that yield believable-looking images.
The goal is that at training time, your model will never see the exact same picture twice.
This helps expose the model to more aspects of the data and generalize better

"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt


# -------------------------------data----------------------------
train_dir = '/home/david/PycharmProjects/datasets/pet_images_data/training_dataset'
validation_dir = '/home/david/PycharmProjects/datasets/pet_images_data/validation_dataset'


#  ----------------- Create de model that includes dropout--------------

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(0.5)) # add dropout to mitigate de overfitting

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

rms = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])

#  --------------- Data preprocessing Using data augmentation ------------------------

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                     target_size=(150, 150),
                                                     batch_size=20,
                                                     class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                     target_size=(150, 150),
                                                     batch_size=20,
                                                     class_mode='binary')



# ------------------------ train the model ---------------------------------

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

# ------------------------ save the model ---------------------------------

model.save('cats_and_dogs_small_2.h5')

# ------------------------ plot the results -------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
