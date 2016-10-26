import sys
import numpy as np
import pandas as pd
import math

def load_test(file):
    testbox = np.genfromtxt(file, delimiter=',')
    testbox = np.delete(testbox, 0, 1)
    return testbox

def test(testbox,P0,P1,mean_0,mean1,sigma):
    test_y = np.ones((len(testbox),1))
    for idx in range(600):
        x = testbox[idx,0:57]
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
        if (p >= 0.5):
            test_y[idx] = 1
        else:
            test_y[idx] = 0
    return test_y

def write_ans1(test_y):
    finalString = "id,label\n"
    for i in range(0,len(test_y)):
        value = int(test_y[i])
        finalString += str(i+1) + "," + str(value) + "\n"
    f = open(prediction, "w")
    f.write(finalString)
    f.close()

model_name = sys.argv[1]
testing_data = sys.argv[2]
prediction = sys.argv[3]

#load model
modelParam = np.genfromtxt(model_name, delimiter=',')
P0 = modelParam[0][0]
P1 = modelParam[1][0]
mean_0 = modelParam[2,:]
mean_1 = modelParam[3,:]
sigma = modelParam[4:62,:]

#testing
testbox = load_test(testing_data)
test_y = test(testbox,P0,P1,mean_0,mean_1,sigma)
write_ans1(test_y)
