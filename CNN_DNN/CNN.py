# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.utils import np_utils

#print("=====  Loading data...  =====")
#data = pd.read_csv('train.csv')
#
#Ytrain = data.iloc[:, 0].values.astype(int)
#Xtrain = data.iloc[:, 1].str.split(expand=True).values.astype(int)
#
#percent = 0.2
#index = np.random.choice(len(Xtrain), int(len(Xtrain)*percent), replace=False)
#Xtest = Xtrain[index]
#Ytest = Ytrain[index]
#
#Xtrain = np.delete(Xtrain, index, 0)
#Ytrain = np.delete(Ytrain, index, 0)
#
#percent = 0.2
#index = np.random.choice(len(Xtrain), int(len(Xtrain)*percent), replace=False)
#Xval = Xtrain[index]
#Yval = Ytrain[index]
#print("=====  Loading Complete  =====")

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], 48, 48, 1))
Xval = np.reshape(Xval, (Xval.shape[0], 48, 48, 1))
Xtest = np.reshape(Xtest, (Xtest.shape[0], 48, 48, 1))

Ytrain = np_utils.to_categorical(Ytrain, 7)

#Build model
input_img = Input(shape=(48, 48, 1))
block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

block2 = Conv2D(64, (3, 3), activation='relu')(block1)
block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

block3 = Conv2D(64, (3, 3), activation='relu')(block2)
block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

block4 = Conv2D(128, (3, 3), activation='relu')(block3)
block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

block5 = Conv2D(128, (3, 3), activation='relu')(block4)
block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
block5 = Flatten()(block5)

fc1 = Dense(1024, activation='relu')(block5)
fc1 = Dropout(0.5)(fc1)

fc2 = Dense(1024, activation='relu')(fc1)
fc2 = Dropout(0.5)(fc2)

predict = Dense(7)(fc2)
predict = Activation('softmax')(predict)
model = Model(inputs=input_img, outputs=predict)

opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

model.fit(Xtrain, Ytrain, epochs=10, batch_size=32)
