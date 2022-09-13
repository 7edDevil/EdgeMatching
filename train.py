import pandas as pd
import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed, Flatten
from sklearn.metrics import accuracy_score, f1_score
import time
import keras
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score







from urllib.request import urlopen,urlretrieve
from PIL import Image
from sklearn.utils import shuffle
import cv2
from resnets_utils import *

import tensorflow._api.v2.compat.v1 as tff

tff.disable_v2_behavior()

from keras.models import load_model
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

















print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# def load_data(dir):
#     data_array = []
#     label_array = []
#     folders = [os.path.join(dir, f) for f in os.listdir(dir)]
#     for folder in folders:
#         files = [os.path.join(folder, g) for g in os.listdir(folder)]
#         for file in files:
#             points = [os.path.join(os.path.join(file, "csv"), p) for p in sorted(os.listdir(os.path.join(file, "csv")))]
#             for j in range(len(points)):
#                 i = j
#                 pre_data = []
#                 while i < j:
#                     data = np.array(json.load(open(points[i]))).astype(float)
#                     if data.shape == (25,1):
#                         pre_data.append(data.tolist())
#                     else:
#                         print("something went wrong with data shape: ", file, data.shape)
#                     i = i + 1
#                 final = np.array(pre_data)
#                 if final.shape == (25,1):
#                     data_array.append(final)
#                     label = folder.split('/')[-1]
#                     label_array.append(label)
#     data_array = np.array(data_array)
#     label_array = np.array(label_array)
#     return data_array, label_array
#





def load_data(dir):
    data_array = []
    label_array = []
    folders = [os.path.join(dir, f) for f in os.listdir(dir)]
    for folder in folders:
        files = [os.path.join(folder, g) for g in os.listdir(folder)]
        for j in range(len(files)):
            i=j
            pre_data = []
            # while True:
            data = np.array(json.load(open(files[j]))).astype(float)
            label = folder.split('\\')[-1]
            #data.append(label)
            #print(np.shape(data))
            if data.shape == (25,):
                pre_data.append(data.tolist())
                if label == 'cane':
                    pre_data.append(0)
                elif label == 'cavallo':
                    pre_data.append(1)
                elif label == 'elefante':
                    pre_data.append(2)
                elif label == 'farfalla':
                    pre_data.append(3)
                elif label == 'gallina':
                    pre_data.append(4)
                elif label == 'gatto':
                    pre_data.append(5)
                elif label == 'mucca':
                    pre_data.append(6)
                elif label == 'pecora':
                    pre_data.append(7)
                elif label == 'ragno':
                    pre_data.append(8)
                elif label == 'scoiattolo':
                    pre_data.append(9)


                #pre_data.append(label)
            else:
                print("something went wrong with data shape: ", files[i], data.shape)
            i = i + 1
            final = np.array(pre_data)
            if True:          #final.shape == (25, 1):
                data_array.append(final)
                label = folder.split('\\')[-1]
                label_array.append(label)
        # label = folder.split('\\')[-1]
        # label_array.append(label)

    data_array = np.array(data_array)
    label_array = np.array(label_array)
    print(type(label_array))
    return data_array, label_array










# define image dataset
# Data Augmentation
image_generator = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2],# brightness
        validation_split=0,)

#Train & Validation Split
train_dataset = image_generator.flow_from_directory(batch_size=64,
                                                 directory=r'D:\University of Surrey\Project\EdgeMatching-master\CuratedSobel_noYOLO',
                                                 shuffle=True,
                                                 target_size=(227,227),
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=64,
                                                 directory=r'D:\University of Surrey\Project\EdgeMatching-master\CuratedSobel_noYOLO',
                                                 shuffle=True,
                                                 target_size=(227,227),
                                                 subset="validation",
                                                 class_mode='categorical')

#Organize data for our predictions
image_generator_submission = ImageDataGenerator(rescale=1/255)
submission = image_generator_submission.flow_from_directory(
                                                directory=r'D:\University of Surrey\Project\EdgeMatching-master\CuratedSobel_noYOLO',
                                                shuffle=False,
                                                target_size=(227,227),
                                                class_mode=None)

#
# train_data, train_labels = load_data(r"D:\University of Surrey\Project\EdgeMatching-master\DatasetJson\ZernikeRawJson_Train")
# val_data, val_labels = load_data(r"D:\University of Surrey\Project\EdgeMatching-master\DatasetJson\ZernikeRawJson_Val")

# train_data=(r"D:\University of Surrey\Project\EdgeMatching-master\Dataset\ZernikeRaw_Train")
# val_data=(r"D:\University of Surrey\Project\EdgeMatching-master\Dataset\ZernikeRaw_Val")

# tf.convert_to_tensor(train_data)
# normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
# normalizer.adapt(train_data)
#print(train_data[0][0][25])

#
# model = keras.models.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = [224, 224,3]),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Conv2D(64, (2, 2), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Conv2D(128, (2, 2), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Conv2D(128, (2, 2), activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(10, activation ='softmax')
# ])

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
#     keras.layers.Dense(10, activation='softmax')
# ])

#
# base_model = tf.keras.applications.ResNet50(weights= None, include_top=False, input_shape= (227,227,3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.7)(x)
# predictions = Dense(10, activation= 'softmax')(x)
# model = Model(inputs = base_model.input, outputs = predictions)
#
# from keras.optimizers import SGD, Adam
# # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# adam = Adam(lr=0.0001)
# model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_dataset, epochs = 100, validation_dataset= validation_dataset)
# preds = model.evaluate(validation_dataset)
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))



# model = keras.models.Sequential([
#     keras.layers.Conv1D(32, 3, padding="same", activation='relu', input_shape=(1,25)),
#     keras.layers.MaxPool1D(),
#     keras.layers.Flatten(),
#     keras.layers.Dense(250, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])


#
# model = Sequential([
#         TimeDistributed(Flatten(input_shape=(1,25))),
#         LSTM(90, return_sequences=False, input_shape=(1,25)),
#         Dense(45, activation=keras.layers.LeakyReLU(alpha=0.01)),
#         Dense(9, activation=keras.layers.LeakyReLU(alpha=0.01)),
#         Dense(10, activation='softmax')
#     ])
#



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




#
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
#              loss = 'categorical_crossentropy',
#              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            restore_best_weights=True)


model_dir = r"D:\University of Surrey\Project\EdgeMatching-master\cnn-model_CuratedSobel_noYOLO_alexretry"
model = keras.models.load_model(model_dir)
print(train_dataset.classes)
from sklearn.metrics import plot_confusion_matrix
Y_pred = model.predict(train_dataset)
print(Y_pred)
loss, accuracy = model.evaluate(train_dataset)
print(accuracy)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train_dataset.classes, y_pred))


# model.fit(train_dataset, epochs=120, validation_data=validation_dataset, callbacks=None)
#model.fit(train_data, train_labels, validation_split=0.1, epochs=100)

# loss, accuracy = model.evaluate(validation_dataset)
# print("Loss: ", loss)
# print("Accuracy: ", accuracy)
#
# model.save('cnn-model_CuratedSobel_noYOLO_alexretry')




#
# nsamples, nx, ny = train_data.shape
# d2_train_data = train_data.reshape((nsamples,nx*ny))
# #
# # nsamples, nx, ny = train_labels.shape
# # d2_train_labels = train_labels.reshape((nsamples,nx*ny))
#
# nsamples, nx, ny = val_data.shape
# d2_val_data = val_data.reshape((nsamples,nx*ny))
# #
# # nsamples, nx, ny = train_data.shape
# # d2_val_labels = val_labels.reshape((nsamples,nx*ny))
#
# #
# # knn = KNeighborsClassifier(n_neighbors=150, p=2,
# #                            metric='minkowski')
# # knn.fit(d2_train_data,train_labels)
# # y_pred = knn.predict(d2_val_data)
#
#
# rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(d2_train_data, train_labels)
# poly = svm.SVC(kernel='poly', degree=3, C=1).fit(d2_train_data, train_labels)
#
# poly_pred = poly.predict(d2_val_data)
# rbf_pred = rbf.predict(d2_val_data)
#
#
# poly_accuracy = accuracy_score(val_labels, poly_pred)
# poly_f1 = f1_score(val_labels, poly_pred, average='weighted')
# print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
# print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
#
# poly_accuracy = accuracy_score(val_labels, poly_pred)
# poly_f1 = f1_score(val_labels, poly_pred, average='weighted')
# print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
# print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

# cm = confusion_matrix(val_labels, y_pred)
# ac = accuracy_score(val_labels, y_pred)
#
# print(ac)