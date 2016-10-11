import sys
import csv
import numpy as np
import pandas as pd
import math
#import matplotlib.pyplot as plt

def load(file):
    Xtrain = pd.read_csv(file)
    featureIdx = range(0,len(Xtrain.columns))
    featureIdx.remove(1)
    Xtrain = Xtrain[featureIdx]
    Xtrain = Xtrain.as_matrix(columns=Xtrain.columns[2:])
    trainbox = np.zeros((240,24))
    for idx in range(9,4329,18):
        trainbox[(idx-9)/18] = Xtrain[idx]
    trainbox = trainbox.reshape(1,24*240)

#    x = range(0,90)
#    y = trainbox[0,0:90]
#    plt.plot(x,y,'b')
#    plt.xlim(0,100)
#    plt.show()  

    num_col = 24*240-9
    box = np.zeros((num_col,9))
    for idx in range(0,num_col):
        box[idx] = trainbox[0,idx:idx+9]
    real_y = np.zeros((num_col,1))
    for idx in range(0,num_col):
        real_y[idx] = trainbox[0,idx+9]
    del trainbox
    return box, real_y  

def train2(box, real_y):
    num_col = 24*240-9
    w = np.ones((9,1))*(0.11)
    b = np.ones((num_col,1))
    rate_w = 0.0001
    rate_b = 0.0001
    x = box.dot(w)
    x = np.add(x,b)
    x0 = box[:,0]
    x0 = x0.reshape(num_col,1)
    x1 = box[:,1]
    x1 = x1.reshape(num_col,1)
    x2 = box[:,2]
    x2 = x2.reshape(num_col,1)
    x3 = box[:,3]
    x3 = x3.reshape(num_col,1)
    x4 = box[:,4]
    x4 = x4.reshape(num_col,1)
    x5 = box[:,5]
    x5 = x5.reshape(num_col,1)
    x6 = box[:,6]
    x6 = x6.reshape(num_col,1)
    x7 = box[:,7]
    x7 = x7.reshape(num_col,1)
    x8 = box[:,8]
    x8 = x8.reshape(num_col,1)
    est_y = np.add(x,b)
    d_Lb = sum((-1)*2*(real_y-est_y))/(len(est_y))
    d_Lw0 = sum((-1)*2*(np.multiply((real_y-est_y), x0)))/(len(est_y))
    d_Lw1 = sum((-1)*2*(np.multiply((real_y-est_y), x1)))/(len(est_y))
    d_Lw2 = sum((-1)*2*(np.multiply((real_y-est_y), x2)))/(len(est_y))
    d_Lw3 = sum((-1)*2*(np.multiply((real_y-est_y), x3)))/(len(est_y))
    d_Lw4 = sum((-1)*2*(np.multiply((real_y-est_y), x4)))/(len(est_y))
    d_Lw5 = sum((-1)*2*(np.multiply((real_y-est_y), x5)))/(len(est_y))
    d_Lw6 = sum((-1)*2*(np.multiply((real_y-est_y), x6)))/(len(est_y))
    d_Lw7 = sum((-1)*2*(np.multiply((real_y-est_y), x7)))/(len(est_y))
    d_Lw8 = sum((-1)*2*(np.multiply((real_y-est_y), x8)))/(len(est_y))
    b = b - rate_b*d_Lb
    w[0] = w[0] - rate_w*d_Lw0
    w[1] = w[1] - rate_w*d_Lw1
    w[2] = w[2] - rate_w*d_Lw2
    w[3] = w[3] - rate_w*d_Lw3
    w[4] = w[4] - rate_w*d_Lw4
    w[5] = w[5] - rate_w*d_Lw5
    w[6] = w[6] - rate_w*d_Lw6
    w[7] = w[7] - rate_w*d_Lw7
    w[8] = w[8] - rate_w*d_Lw8
    new_est_y = np.add(box.dot(w),b)
    new_L = sum(np.sqrt((real_y-new_est_y)**2))/(len(est_y))

    for i in range(10000):
        w[0] = w[0] - rate_w*d_Lw0
        w[1] = w[1] - rate_w*d_Lw1
        w[2] = w[2] - rate_w*d_Lw2
        w[3] = w[3] - rate_w*d_Lw3
        w[4] = w[4] - rate_w*d_Lw4
        w[5] = w[5] - rate_w*d_Lw5
        w[6] = w[6] - rate_w*d_Lw6
        w[7] = w[7] - rate_w*d_Lw7
        w[8] = w[8] - rate_w*d_Lw8
        new_est_y = np.add(box.dot(w),b)
        d_Lw0 = sum((-1)*2*(np.multiply((real_y-new_est_y), x0)))/(len(est_y))
        d_Lw1 = sum((-1)*2*(np.multiply((real_y-new_est_y), x1)))/(len(est_y))
        d_Lw2 = sum((-1)*2*(np.multiply((real_y-new_est_y), x2)))/(len(est_y))
        d_Lw3 = sum((-1)*2*(np.multiply((real_y-new_est_y), x3)))/(len(est_y))
        d_Lw4 = sum((-1)*2*(np.multiply((real_y-new_est_y), x4)))/(len(est_y))
        d_Lw5 = sum((-1)*2*(np.multiply((real_y-new_est_y), x5)))/(len(est_y))
        d_Lw6 = sum((-1)*2*(np.multiply((real_y-new_est_y), x6)))/(len(est_y))
        d_Lw7 = sum((-1)*2*(np.multiply((real_y-new_est_y), x7)))/(len(est_y))
        d_Lw8 = sum((-1)*2*(np.multiply((real_y-new_est_y), x8)))/(len(est_y))
        L = sum(np.sqrt((real_y-new_est_y)**2))/(len(est_y))
    print L,sum(b)/len(b)
    return w, b
        
def load_test(file):
    Xtrain = pd.read_csv(file)
    featureIdx = range(0,len(Xtrain.columns))
    Xtrain = Xtrain[featureIdx]
    # convert pd dataframe to np array
    Xtrain = Xtrain.as_matrix(columns=Xtrain.columns[2:])
    testbox = np.zeros((240,9))
    for idx in range(8,4328,18):
        testbox[(idx-8)/18] = Xtrain[idx]
    return testbox

def test(testbox,w,b):
    w = w.reshape(9,1)
    x = testbox.dot(w)
    print len(x)
    b = np.ones((len(x),1))  # b = 1
    # start test
    test_y = np.add(x,b)
    for x in range(240):
        test_y[x] = round(test_y[x])
    return test_y

def write_ans(test_y):
    finalString = "id,value\n"
    for i in range(240):
        value = test_y[i]
        finalString += "id_" + str(i) + "," + str(round(value)) + "\n"
    f = open('kaggle_best.csv', "w")
    f.write(finalString)
    f.close()

#training
[box, real_y] = load('train.csv')
[w, b] = train2(box, real_y)

#testing
testbox = load_test('test_X.csv')
test_y = test(testbox,w,b)
write_ans(test_y)




