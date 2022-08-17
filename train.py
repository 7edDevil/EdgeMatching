import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
import time
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#define image dataset
# Data Augmentation
image_generator = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2],# brightness
        validation_split=0.2,)

#Train & Validation Split
train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='/home/dheeraj/Downloads/dataset/dataset',
                                                 shuffle=True,
                                                 target_size=(227, 227),
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='/home/dheeraj/Downloads/dataset/dataset',
                                                 shuffle=True,
                                                 target_size=(227, 227),
                                                 subset="validation",
                                                 class_mode='categorical')

#Organize data for our predictions
image_generator_submission = ImageDataGenerator(rescale=1/255)
submission = image_generator_submission.flow_from_directory(
                                                 directory='/home/dheeraj/Downloads/dataset/dataset',
                                                 shuffle=False,
                                                 target_size=(227, 227),
                                                 class_mode=None)



model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = [224, 224,3]),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (2, 2), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, (2, 2), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, (2, 2), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(151, activation ='softmax')
])

#
# model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(151, activation='softmax')
# ])
#
# root_logdir = os.path.join(os.curdir, "logs\\fit\\")
# def get_run_logdir():
#     run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#     return os.path.join(root_logdir, run_id)
# run_logdir = get_run_logdir()
# tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
#
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
# model.summary()
#
# model.fit(train_dataset, epochs=50, validation_data=validation_dataset, validation_freq=1, callbacks=[tensorboard_cb])
#





model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            restore_best_weights=True)


model.fit(train_dataset, epochs=50, validation_data=validation_dataset, callbacks=callback)


loss, accuracy = model.evaluate(validation_dataset)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

model.save('cnn-model??')
