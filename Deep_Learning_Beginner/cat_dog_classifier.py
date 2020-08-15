#!usr/bin/env python3

"""
Date: 06 - April - 2020
Description: Based on chapter 5, using convnets with small datasets
section 5.2
This is an image classification program using convnets,
it will focus on classifying images as dogs or cats

Dataset: 25_000 pictures --> 12_500 dogs, 12_500 cats --> stored in pet_dog_data folder

For this example I am only using:

         training dataset   = 2000 images
         validation dataset = 1000 images
         testing dataset    = 1000 image

I am using this small date in order to understand and using some concepts

El resultado de este programa son dos graficas donde se ve claramente el overfitting
es decir un buen rendimiento con el training data pero un mal rendimiento con el validation date.
En base a esto es necesario aplicar alguna tecnicas para disminuir el overffit. eso se hace en los siguientes
codigos


"""

import os
import shutil
import piexif  # special library to fix a problem related with the images (I added this)
import tensorflow as tf
import matplotlib.pyplot as plt

original_dataset_directory = '/home/david/PycharmProjects/datasets/pet_images_data/'

# -------- Create the directories where I will copy the image for train, test and validation---------

train_dir = os.path.join(original_dataset_directory, 'training_dataset')
test_dir = os.path.join(original_dataset_directory, 'testing_dataset')
validation_dir = os.path.join(original_dataset_directory, 'validation_dataset')

train_dir_dog = os.path.join(train_dir, 'training_Dog')
train_dir_cat = os.path.join(train_dir, 'training_Cat')
test_dir_dog = os.path.join(test_dir, 'test_Dog')
test_dir_cat = os.path.join(test_dir, 'test_Cat')
vali_dir_dog = os.path.join(validation_dir, 'validation_Dog')
vali_dir_cat = os.path.join(validation_dir, 'validation_Cat')

try:
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(validation_dir)
    os.mkdir(train_dir_dog)
    os.mkdir(train_dir_cat)
    os.mkdir(test_dir_dog)
    os.mkdir(test_dir_cat)
    os.mkdir(vali_dir_dog)
    os.mkdir(vali_dir_cat)
except:
    pass

# --------- Copying images to training, test, and validation directories -------------

CATEGORIES = ["Dog", "Cat"]
for category in CATEGORIES:
    path = os.path.join(original_dataset_directory, category)  # go a directory to dogs and cats

    for i in range(1_000):
        # For each category take the first 1000 images and copy them to the respective train_directory
        src = os.path.join(path, '{}.jpg'.format(i))

        piexif.remove(src)  # I added this line in order to eliminate a error related to Exif files in the images

        dst = os.path.join(train_dir, 'training_{}/{}.jpg'.format(category, i))
        shutil.copyfile(src, dst)

    for i in range(1_000, 1_500):
        # For each category take the next 500 images and copy them to the respective test_directory
        src = os.path.join(path, '{}.jpg'.format(i))
        piexif.remove(src)
        dst = os.path.join(test_dir, 'test_{}/{}.jpg'.format(category, i))
        shutil.copyfile(src, dst)

    for i in range(1_500,  2_000):
        # For each category take the next 500 images and copy them to the respective validation_directory
        src = os.path.join(path, '{}.jpg'.format(i))
        piexif.remove(src)
        dst = os.path.join(validation_dir, 'validation_{}/{}.jpg'.format(category, i))
        shutil.copyfile(src, dst)

print("total training cat images:", len(os.listdir(train_dir_cat)))
print("total training dog images:", len(os.listdir(train_dir_dog)))

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
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# ------------------------ train the model ---------------------------------

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

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
