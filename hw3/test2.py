from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout,UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, sgd
from keras.utils import np_utils
import sys
import numpy as np
import pickle, theano
from itertools import chain, repeat


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


def write_ans(result,name):
    ans = np.zeros((len(result), 1))
    for i in range(len(result)):
        for j in range(10):
            if result[i, j] > 0.5:
                ans[i] = j

    finalString = "ID,class\n"
    for i in range(len(ans)):
        value = int(ans[i])
        finalString += str(i) + "," + str(value) + "\n"
    f = open(name, "w")
    f.write(finalString)
    f.close()

path_name = sys.argv[1]
model_name = sys.argv[2]
prediction = sys.argv[3]

test = load_test(path_name + '/test.p')
model = load_model(model_name)

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
# X_val, Y_val = validation(X_train, Y_train)

autoencoder.fit(X_train.reshape(X_train.shape[0], 3, 32, 32), X_train.reshape(X_train.shape[0], 3, 32, 32),
                nb_epoch=30,
                batch_size=32,
                shuffle=True)

Xtrain = encoder(test.reshape(test.shape[0], 3, 32, 32))
Xtrain = np.asarray(Xtrain[0])
Xtrain = Xtrain.astype('float32')

result = model.predict(Xtrain, batch_size=32, verbose=1)

write_ans(result, prediction)
