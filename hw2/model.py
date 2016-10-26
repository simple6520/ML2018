import sys
import numpy as np
import pandas as pd
import math
#import matplotlib.pyplot as plt

def load(file):
    Xtrain = np.genfromtxt(file, delimiter=',')
    ans = Xtrain[:,58]
    ans = ans.reshape(4001,1)
    Xtrain = np.delete(Xtrain, 58, 1) #delete answer
    Xtrain = np.delete(Xtrain, 0, 1)
    return Xtrain, ans

def feature_scaling(box):
    m = box.sum(axis=0)
    m = m / 4001
    m = np.array(m)
    DV = np.ones((len(box[0]),1))*0
    for idx in range(len(box[0])):
        DV[idx] = np.sqrt(sum((box[:,idx] - m[idx]) ** 2) / len(box))
        box[:,idx] = (box[:,idx] - m[idx])/DV[idx]
    return box, m, DV
        
def train(box,y):
    ff = np.ones((len(y),1))*0
    rate = 0.005
    w = np.ones((57,1))*0.00001 #average
    b = 0.001
    z = box.dot(w) + b
    f = 1 /(1 + np.exp((-1)*z))
    dw = np.ones((57,1))
    for idx in range(2):
        x = box[:,idx]
        x = x.reshape(len(y),1)
        dL = (-1)*np.multiply((y-f),x)
        dw[idx] = sum(dL)
    db = sum((-1)*(np.add(y,-1*(f))))
    b = b - rate * db
    w = w - rate * dw
    new_z = box.dot(w) + b
    new_f = 1 /(1 + np.exp((-1)*new_z))
    for i in range(6):
        for idx in range(57):
            x = box[:,idx]
            x = x.reshape(len(y),1)
            dL = (-1)*np.multiply((y-new_f),x)
            dw[idx] = sum(dL)
        if i>5:
            rate = 0.000001
        db = sum((-1)*(np.add(y,-1*(new_f))))
        b = b - rate * db
        w = w - rate * dw
        new_z = box.dot(w) + b
        new_f = 1 /(1 + np.exp((-1)*new_z))
        u = 0
        v = 0
        for j in range(len(new_f)):
            if new_f[j] >= 0.5:
                u = u + y[j]*np.log(new_f[j])
            else:
                v = v + (1-y[j])*np.log(1-new_f[j])
        new_L = -1*sum(u+v)
    return w, b

filename = sys.argv[1]
outputName = sys.argv[2]

# training
[box, ans] = load(filename)
[box, m, DV] = feature_scaling(box)
[w, b] = train(box,ans)

b = np.ones((57,1))*b
np.savetxt(outputName,np.column_stack((w,b,m,DV)),delimiter=',')
