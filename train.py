from os import listdir

import matplotlib.pyplot as plt
import cv2
import numpy as np
# import tensorflow as tf


# Read data
data_folder = 'floorplans16-01'
input_width = 512
input_height = 256

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_folder,
#     validation_split=0.2,
#     subset='training',
#     seed=123,
#     image_size=(input_height, input_width),
# )

# Read grayscale images
filled_floorplans = []
for filename in listdir(data_folder):
  if filename.endswith('.tiff'):
    img = cv2.imread(data_folder + '/' + filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (input_width, input_height))
    filled_floorplans.append(img)

    # Add flipped versions along x axis, y axis and both
    flipped_img = cv2.flip(img, 0)
    filled_floorplans.append(flipped_img)

    flipped_img = cv2.flip(img, 1)
    filled_floorplans.append(flipped_img)

    flipped_img = cv2.flip(img, -1)
    filled_floorplans.append(flipped_img)


empty_floorplan = cv2.imread('dataset/empty.tiff', cv2.IMREAD_GRAYSCALE)
empty_floorplan = cv2.resize(empty_floorplan, (input_width, input_height))


# plt.imshow(filled_floorplans[0])
# plt.show()


# Create a variational autoencoder (VAE) model.
# The encoder part of the VAE is a deep convolutional neural network (CNN)
# The decoder part of the VAE is also a CNN

def train():
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
  from tensorflow.keras.callbacks import TensorBoard

  input_layer = Input(shape=(input_width, input_height, 1))

  x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)

  print('encoded shape:', encoded.shape)

  x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

  print("hi")
  print(decoded.shape)

  autoencoder = Model(input_layer, decoded)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

  # Filled floor plans
  y_data = np.array(filled_floorplans)
  y_data = y_data.astype('float32') / 255.
  y_data = np.reshape(y_data, (len(y_data), input_width, input_height, 1))

  # Empty floor plans
  x_data = np.array([empty_floorplan for _ in range(len(filled_floorplans))])
  x_data = x_data.astype('float32') / 255.
  x_data = np.reshape(x_data, (len(x_data), input_width, input_height, 1))

  split_ratio = 0.8

  y_train = y_data[:int(len(y_data) * split_ratio)]
  y_test = y_data[int(len(y_data) * split_ratio):]

  x_train = x_data[:int(len(x_data) * split_ratio)]
  x_test = x_data[int(len(x_data) * split_ratio):]


  autoencoder.fit(x_train, y_train,
                  epochs=50,
                  batch_size=128,
                  shuffle=True,
                  validation_data=(x_test, y_test),
                  callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

  # Save the model
  autoencoder.save('autoencoder.h5')

  # Plot the results
  decoded_imgs = autoencoder.predict(x_test)
  n = 10
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(input_width, input_height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(input_width, input_height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  plt.show()

train()