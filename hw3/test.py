from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, sgd
from keras.utils import np_utils
import sys
import numpy as np
import pickle
from itertools import chain, repeat

def load_test(file):
    test_data = pickle.load(open(file, 'rb'))

    test = np.zeros((10000, 3, 32, 32))

    for i in range(10000):
        a = np.zeros((1, 3072))
        for j in range(3072):
            a[0, j] = test_data['data'][i][j]
            test[i, :, :, :] = a.reshape((3, 32, 32))

    test = test.astype('float64')
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

result = model.predict(test, batch_size=32, verbose=1)
write_ans(result,prediction)
