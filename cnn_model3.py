# Import libraries
# import pandas as pd
import pickle

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

# Ref: https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# a list of all the labels we have
#
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
          'X', 'Y']
num_of_image = 500  # we have 100 images for each label
image_resolution = 128
input_dir_path = './asl_gray/'

X_train = []
Y_train = []
# read grayscaled images
for i in range(len(labels)):
    for j in range(num_of_image):
        temp_im = cv2.imread(input_dir_path + labels[i] + '/' + labels[i] + str(j) + '.png', cv2.IMREAD_COLOR)
        # temp_im = 255*temp_im/np.max(temp_im)
        temp_im = cv2.resize(temp_im, (image_resolution, image_resolution))
        X_train.append(temp_im)
        Y_train.append(i)


# convert to a 1200x24 matrix
# where if label = i, then the row have 1 in ith column, and 0 otherwise
Y_train = to_categorical(Y_train, num_classes=24)

# normalize the data
X_train = np.array(X_train) / 255
X_train = X_train.reshape(-1, image_resolution, image_resolution, 3)
# print(X_train[0].shape)
# cross-validation: 9:1
X_t, X_v, Y_t, Y_v = train_test_split(X_train, Y_train, test_size=0.3, random_state=0, shuffle=True)

# The model
model = Sequential()
# Convolution layers
# Layer 1
model.add(
    Conv2D(filters=32, kernel_size=(7, 7), padding='Same', activation='relu', input_shape=(image_resolution, image_resolution, 3), use_bias=True))
model.add(MaxPool2D(pool_size=(2, 2)))  # downsampling
model.add(Dropout(0.25))  # Dropout reduces overfitting
# Layer 2
model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='Same', activation='relu', use_bias=True, strides=(2, 2)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Layer 3
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu', use_bias=True))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
#
# # Layer 4
# model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu',use_bias=True))
# model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(24, activation='softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# use categorical crossentropy as loss function
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define model parameters
epochs = 30
batch_size = 40
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.summary()

# Fit model - Use first line when predicting and second "final" line when using validation set to find # of epochs

model.fit(X_t, Y_t, batch_size=batch_size, epochs=epochs, validation_data=(X_v,Y_v), callbacks=[learning_rate_reduction])
# final = model.fit(X_train2, Y_train2, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val), callbacks=[learning_rate_reduction])

json_mod = model.to_json()
with open('cnn_model_500im_epoch30_batch40_layer3_RGB_reshape.json', 'w') as file:
    file.write(json_mod)
model.save_weights("RBGmodel.h5")
print("Saved")

model.evaluate(X_t, Y_t)

"""

test_path = './asl_alphabet_test/'


for lb in labels:
    x = cv2.imread(test_path + lb +'_test.jpg')
    q = cv2.imread('input.png')
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # img_expanded = x[:, :, np.newaxis]
    #x.reshape(200, 200, 1)
    # x = np.expand_dims(img_expanded, axis=0)
    print(x.shape)
    x = cv2.resize(x, (128, 128))
    # x = np.resize()
    y = model.predict_classes(x)
    print(y)
"""
# # save the model to disk
# filename = './finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))


"""Results:
600 images for each class
epoch = 30
batch size = 60
4 conv layers

Epoch 25/30
216/216 [==============================] - 5s 22ms/step - loss: 0.0974 - accuracy: 0.9676 - val_loss: 0.0651 - val_accuracy: 0.9736
Epoch 26/30
216/216 [==============================] - 5s 22ms/step - loss: 0.1011 - accuracy: 0.9693 - val_loss: 0.0182 - val_accuracy: 0.9944
Epoch 27/30
216/216 [==============================] - 5s 22ms/step - loss: 0.0815 - accuracy: 0.9740 - val_loss: 0.0100 - val_accuracy: 0.9986
Epoch 28/30
216/216 [==============================] - 5s 22ms/step - loss: 0.0876 - accuracy: 0.9680 - val_loss: 0.0183 - val_accuracy: 0.9937
Epoch 29/30
216/216 [==============================] - 5s 22ms/step - loss: 0.0812 - accuracy: 0.9745 - val_loss: 0.0161 - val_accuracy: 0.9931
Epoch 30/30
216/216 [==============================] - 5s 22ms/step - loss: 0.0790 - accuracy: 0.9745 - val_loss: 0.0366 - val_accuracy: 0.9924
Saved
405/405 [==============================] - 3s 7ms/step - loss: 0.0085 - accuracy: 0.9978
"""

"""
600 images for each class
epoch = 30
batch size = 60
3 conv layers

Epoch 25/30
216/216 [==============================] - 5s 24ms/step - loss: 0.0760 - accuracy: 0.9733 - val_loss: 0.0269 - val_accuracy: 0.9924
Epoch 26/30
216/216 [==============================] - 5s 21ms/step - loss: 0.0842 - accuracy: 0.9731 - val_loss: 0.0177 - val_accuracy: 0.9951
Epoch 27/30
216/216 [==============================] - 5s 21ms/step - loss: 0.0749 - accuracy: 0.9752 - val_loss: 0.0206 - val_accuracy: 0.9944
Epoch 28/30
216/216 [==============================] - 6s 28ms/step - loss: 0.0714 - accuracy: 0.9767 - val_loss: 0.0213 - val_accuracy: 0.9931
Epoch 29/30
216/216 [==============================] - 6s 28ms/step - loss: 0.0629 - accuracy: 0.9809 - val_loss: 0.0166 - val_accuracy: 0.9944
Epoch 30/30
216/216 [==============================] - 6s 29ms/step - loss: 0.0646 - accuracy: 0.9789 - val_loss: 0.0246 - val_accuracy: 0.9937
Saved
405/405 [==============================] - 3s 8ms/step - loss: 0.0030 - accuracy: 0.9996
"""



