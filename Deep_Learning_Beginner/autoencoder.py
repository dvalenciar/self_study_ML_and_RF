
'''
Description: AutoEncoder implemented with keras and python.
             Here I used a CNN , first encode the image and decode the image
             This is the basic encode, let's note that here I do not use a sequential model
             model instance I  pass each input to each layer

Date: 17 / July /2020

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the data set MNIST
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# Reshape to (28, 28, 1)
train_images = np.reshape(train_images, [len(train_images), 28, 28, 1])
test_images = np.reshape(test_images, [len(test_images), 28, 28, 1])

# Normalize the input images
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Build the Encoder model
input_img = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# Build the decoder model
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)

model = tf.keras.Model(input_img, decoded)  # here turns an input tensor and output tensor into a model
model.summary()
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(train_images, train_images, shuffle=True, epochs=50, batch_size=64, verbose=1)
model.save('basic_auto_encoder.h5')

'''
# Build the Encoder model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

# Build the decoder model
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same'))
print(model.summary())
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_images, epochs=50, batch_size=128, verbose=1)
test_loss, test_accuracy = model.evaluate(test_images, test_images)
'''

def show_imgs(x_test, decoded_imgs=None, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+ 1 +n)
            plt.imshow(decoded_imgs[i].reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

# to test purpose
image_decode = model.predict(test_images)
show_imgs(test_images, image_decode)

