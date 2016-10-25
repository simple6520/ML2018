import sys
import numpy as np
import pandas as pd
import math

def load_test(file,m,DV):
    testbox = np.genfromtxt(file, delimiter=',')
    testbox = np.delete(testbox, 0, 1)
    #feature_scaling
    for idx in range(len(testbox[0])):
        testbox[:,idx] = (testbox[:,idx] - m[idx])/DV[idx]
    return testbox

def test(testbox,w,b):
    f = testbox.dot(w) + b
    # start test
    test_y = 1 /(1 + np.exp((-1)*f))
    for x in range(0,len(test_y)):
        if (test_y[x] >= 0.5):
            test_y[x] = 1
        else:
            test_y[x] = 0
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

modelParam = np.genfromtxt(model_name, delimiter=',')
w = modelParam[:,0]
b = modelParam[0][1]
m = modelParam[:,2]
DV = modelParam[:,3]

#testing
testbox = load_test(testing_data,m,DV)
test_y = test(testbox,w,b)
write_ans1(test_y)
