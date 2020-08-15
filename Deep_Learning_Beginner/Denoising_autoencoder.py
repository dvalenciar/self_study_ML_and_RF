'''
Description: Denoising autoencoder  implemented with keras and python.
            Data set MNIST, CNN architecture
            Train a denoising autoencoder by adding random noise BEFORE training!
            The goal here is to remove the noise from a Image
Date: 18 / July /2020

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

# Function to plot
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


# Add random noise before training!
noise_factor = 0.5
x_train_noisy = train_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
x_test_noisy = test_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
show_imgs(x_test_noisy) # plot some images to see the noise


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
model.fit(x_train_noisy, train_images, shuffle=True, epochs=50, batch_size=64, verbose=1)
model.save('denoising_autoencoder.h5')

image_decode = model.predict(x_test_noisy)
show_imgs(x_test_noisy, image_decode)

