import numpy as np
import pandas as pd

def load_data(file):
    # load data and & One-hot encoding
    Xtrain = pd.read_csv(file, skipinitialspace=True)
    
    df = pd.get_dummies(Xtrain['workclass'])
    Xtrain = Xtrain.drop('workclass', 1)
    Xtrain = pd.concat([df, Xtrain], axis=1)
    df = pd.get_dummies(Xtrain['sex'])
    Xtrain = Xtrain.drop('sex', 1)
    Xtrain = pd.concat([df, Xtrain], axis=1)
    df = pd.get_dummies(Xtrain['marital_status'])
    Xtrain = Xtrain.drop('marital_status', 1)
    Xtrain = pd.concat([df, Xtrain], axis=1)
    df = pd.get_dummies(Xtrain['occupation'])
    Xtrain = Xtrain.drop('occupation', 1)
    Xtrain = pd.concat([df, Xtrain], axis=1)
    Xtrain = Xtrain.drop('relationship', 1)
    
    edu_mapping = {'Doctorate':15, 'Prof-school':15, 'Assoc-voc':10, 'Masters':10, 'Bachelors':5, 'Assoc-acdm':5, 'Some-college':5,
              'HS-grad':0, '12th':0, '11th':0, '10th':0, '9th':0, '7th-8th':0, '5th-6th':0, '1st-4th':0, 'Preschool':0}
    Xtrain['education'] = Xtrain['education'].map(edu_mapping)
    race_mapping = {'White':3,'Black':2,'Asian-Pac-Islander':1,'Amer-Indian-Eskimo':0,'Other':0}
    Xtrain['race'] = Xtrain['race'].map(race_mapping)
    
    # label
    if 'income' in Xtrain:
    income_mapping = {'<=50K':0, '>50K':1}
    Xtrain['income'] = Xtrain['income'].map(income_mapping)

    Xtrain = np.array(Xtrain.values)
    
    return Xtrain
