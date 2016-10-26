#Probabilistic Generative Model 
import sys
import numpy as np
import pandas as pd
import math

def load(file):
    Xtrain = np.genfromtxt(file, delimiter=',')
    ans = Xtrain[:,58]
    ans = ans.reshape(4001,1)
    #Xtrain = np.delete(Xtrain, 58, 1) #delete answer
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
    P0 = (len(y)-sum(y))/len(y)
    P1 = sum(y)/len(y)
    num = sum(box[:,57]) # num of 1
    class_0 = np.ones((4001-num,57))*0
    class_1 = np.ones((num,57))*0
    count0 = 0
    count1 = 0
    for idx in range(4001):
        if box[idx,57] == 0:
            class_0[count0] = box[idx,0:57]
            count0 = count0 + 1
        else:
            class_1[count1] = box[idx,0:57]
            count1 = count1 + 1
    mean_0 = class_0.sum(axis=0)/len(class_0)
    mean_1 = class_1.sum(axis=0)//len(class_1)
    c0 = 0
    c1 = 0
    for idx in range(len(class_0)):
        a0 = class_0[idx] - mean_0
        b0 = a0.reshape(57,1)
        c = b0*a0
        c0 = c0 + c
    for idx in range(len(class_1)):
        a1 = class_1[idx] - mean_1
        b1 = a1.reshape(57,1)
        d = b1*a1 
        c1 = c1 + d
    sigma0 = c0 / len(class_0)
    sigma1 = c1 / len(class_1)
    sigma = (sigma0*len(class_0)+sigma1*len(class_1))/ (len(class_0)+len(class_1))

    #testing
    for idx in range(1):
        x = box[idx,0:57]
        A0 = x-mean_0
        A0t = A0.reshape(57,1)
        b0 = np.linalg.inv(sigma)
        R0 = A0.dot(b0)
        R0 = R0.dot(A0t)
        g0 = (1/((2*np.pi)**(57/2))) * (1/np.sqrt(np.linalg.det(sigma))) * (np.exp(-0.5*R0)) 

        A1 = x-mean_1
        A1t = A1.reshape(57,1)
        b1 = np.linalg.inv(sigma)
        R1 = A1.dot(b1)
        R1 = R1.dot(A1t)
        g1 = (1/((2*np.pi)**(57/2))) * (1/np.sqrt(np.linalg.det(sigma))) * (np.exp(-0.5*R1)) 

        p = g1*P1 / (g0*P0+g1*P1)

    return P0,P1,mean_0,mean_1,sigma


filename = sys.argv[1]
outputName = sys.argv[2]

# training
[box, ans] = load(filename)
[P0,P1,mean_0,mean_1,sigma] = train(box,ans)

P0 = np.ones((1,57))*P0
P1 = np.ones((1,57))*P1

# output model
np.savetxt(outputName,np.vstack((P0,P1,mean_0,mean_1,sigma)),delimiter=',')
