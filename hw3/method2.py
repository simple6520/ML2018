from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D,UpSampling2D
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
import pickle
import theano
import sys
from itertools import chain, repeat


def write_ans(result):
    ans = np.zeros((len(result), 1))
    for i in range(len(result)):
        for j in range(10):
            if result[i, j] > 0.5:
                ans[i] = j

    finalString = "ID,class\n"
    for i in range(len(ans)):
        value = int(ans[i])
        finalString += str(i) + "," + str(value) + "\n"
    f = open('out', "w")
    f.write(finalString)
    f.close()


def write_ans1(result):
    ans = np.zeros((len(result), 1))
    for i in range(len(result)):
        for j in range(10):
            if result[i, j] > 0.5:
                ans[i] = j

    finalString = "ID,class\n"
    for i in range(len(ans)):
        value = int(ans[i])
        finalString += str(i) + "," + str(value) + "\n"
    f = open('prediction', "w")
    f.write(finalString)
    f.close()


def load_label(file):
    label_data = pickle.load(open(file, 'rb'))
    pic = list()
    for i in range(10):
        for j in range(500):
            pic.append(label_data[i][j])

    Xtrain_label = np.asarray(pic)
    X_train = Xtrain_label.astype('float32')
    X_train /= 255

    Y_temp = list(range(10))
    Y_train = list(chain.from_iterable(repeat(e, 500) for e in Y_temp))
    Y_train = np.asarray(Y_train)
    Y_train = np_utils.to_categorical(Y_train, 10)


    index = [i * 5 for i in range(1000)]
    new_X = np.delete(X_train, index, 0)  # training set
    new_Y = np.delete(Y_train, index, 0)  # ans of training set

    return new_X, new_Y


def load_unlabel(file):
    unlabel_data = pickle.load(open(file, 'rb'))
    unlabel_data = np.asarray(unlabel_data)

    X_train = unlabel_data.astype('float32')
    X_train /= 255
    return X_train


def load_test(file):
    test_data = pickle.load(open(file, 'rb'))

    test = np.zeros((10000, 3, 32, 32))

    for i in range(10000):
        a = np.zeros((1, 3072))
        for j in range(3072):
            a[0, j] = test_data['data'][i][j]
            test[i, :, :, :] = a.reshape((3, 32, 32))

    test = test.astype('float32')
    test /= 255
    return test


def validation(Xtrain, Ytrain):
    X_val = np.zeros((1000, 3072))
    Y_val = np.zeros((1000, 10))
    for i in range(len(Xtrain)/5):
        X_val[i][:] = Xtrain[i*5][:]
        Y_val[i][:] = Ytrain[i*5][:]

    # X_val = X_val.reshape(X_val.shape[0], 3, 32, 32)
    return X_val, Y_val

path_name = sys.argv[1]
model_name = sys.argv[2]


# encoder
autoencoder = Sequential()
autoencoder.add(Convolution2D(3, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
autoencoder.add(MaxPooling2D((2, 2)))
autoencoder.add(Activation('relu'))
autoencoder.add(Convolution2D(32, 3, 3, border_mode='same'))
autoencoder.add(MaxPooling2D((2, 2)))
autoencoder.add(Activation('relu'))
x = MaxPooling2D((2, 2))
autoencoder.add(x)
encoder = theano.function([autoencoder.input], [x.output])

# decoder
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Activation('relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Convolution2D(32, 3, 3, border_mode='same'))
autoencoder.add(Activation('sigmoid'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Convolution2D(3, 3, 3, border_mode='same'))


autoencoder.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['mse'])


X_train, Y_train = load_label(path_name + '/all_label.p')
X_unlabel = load_unlabel(path_name + 'all_unlabel.p')
# X_val, Y_val = validation(X_train, Y_train)

autoencoder.fit(X_train.reshape(X_train.shape[0], 3, 32, 32), X_train.reshape(X_train.shape[0], 3, 32, 32),
                nb_epoch=20,
                batch_size=32,
                shuffle=True)

# autoencoder.save(model_name1)


Xtrain = encoder(X_train.reshape(X_train.shape[0], 3, 32, 32))
Xtrain = np.asarray(Xtrain[0])

Xunlabel = encoder(X_unlabel.reshape(X_unlabel.shape[0], 3, 32, 32))
Xunlabel = np.asarray(Xunlabel[0])


# X_val = X_val.astype('float32')
# X_val = encoder(X_val.reshape(X_val.shape[0], 3, 32, 32))
# X_val = np.asarray(X_val[0])

# model
model = Sequential()
model.add(UpSampling2D((2, 2),input_shape=(32, 4, 4)))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('sigmoid'))
model.add(UpSampling2D((2, 2)))
model.add(Convolution2D(3, 3, 3, border_mode='same'))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(Xtrain, Y_train,
                nb_epoch=10,
                batch_size=32,
                shuffle=True)

Y_unlabel = model.predict(Xunlabel, batch_size=32, verbose=1)

X = np.concatenate((Xtrain, Xunlabel), axis=0)
Y = np.concatenate((Y_train, Y_unlabel), axis=0)

model.fit(X, Y,
          nb_epoch=1,
          batch_size=32,
          shuffle=True)

model.save(model_name)









