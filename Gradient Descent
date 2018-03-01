import pandas as pd
import numpy as np


def write_ans(test_y):
    finalString = "id,value\n"
    for i in range(240):
        value = test_y[i]
        finalString += "id_" + str(i) + "," + str(int(round(value))) + "\n"
    f = open('linear_regression.csv', "w")
    f.write(finalString)
    f.close()


def feature_scaling(box):
    m = box.sum(axis=0)
    m = m / box.shape[0]
    m = np.array(m)
    DV = np.ones((len(box[0]), 1))*0
    for idx in range(len(box[0])):
        DV[idx] = np.sqrt(sum((box[:, idx] - m[idx]) ** 2) / len(box))
        box[:, idx] = (box[:, idx] - m[idx])/DV[idx]
    return box, m, DV


para_num = 3
"""  load training data  """""""""""""""
data = pd.read_csv('train.csv', sep=',', header=None)
df1 = np.asarray(data)

start = False
for x in range(df1.shape[0]):
    if df1[x, 2] == 'PM2.5':
        if not start:
            df2 = df1[x, 3:]
            start = True
        else:
            a = df1[x, 3:]
            df2 = np.hstack([df2, a])
y = np.zeros(shape=(df2.shape[0]-9))
df2 = map(int, df2)
real_y = np.asarray([df2[9:]]).T
""""""""""""""""""""""""""""""""""""""""""

"""  load testing data  """""""""""""""
data2 = pd.read_csv('test_X.csv', sep=',', header=None)
data2 = np.asarray(data2)

start = False
for x in range(data2.shape[0]):
    if data2[x, 1] == 'PM2.5':
        if not start:
            test = map(int, data2[x, 2:])
            start = True
        else:
            a = map(int, data2[x, 2:])
            test = np.vstack([test, a])
""""""""""""""""""""""""""""""""""""""""""""

w = np.ones((9, 1))*0.01
rate = 0.0001
r = 0.1

box = np.empty((0, 9))
for x in range(len(df2)-9):
    Input = np.array([df2[x], df2[x+1], df2[x+2], df2[x+3], df2[x+4], df2[x+5], df2[x+6], df2[x+7], df2[x+8]])
    box = np.vstack([box, Input])



y = np.dot(box, w)  # First estimate

temp = 10000
for x in range(20000):
    d_Lw = (-1) * 2 * (np.dot((real_y-y).T, box)) / (len(y)) + r*sum(w ** 2)
    w -= rate * d_Lw.T
    y = np.dot(box, w)
    L = sum(np.abs(real_y-y))/len(y)
    if L < temp:
        best_w = w
        temp = L
    print L, x

print temp
ans = np.dot(test, best_w)
for x in range(len(ans)):
    if ans[x] <= 0:
        ans[x] = 0

write_ans(ans)

