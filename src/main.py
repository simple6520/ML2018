import pandas as pd
import numpy as np
import IDS_utility as ids_util
import IDS_feature_extract as ids_fe
import IDS_learning_model as ids_lm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier

""" Loading training and testing data """
print "==============Loading training and testing data...=============="
train_file='data/train'
test_file='data/test'
type_file='data/training_attack_types.txt'

test_data=pd.read_csv(test_file, delimiter=",", header=None)
train_data_feature,y_train=ids_util.load_training(train_file,type_file,train_le=False)

train_data_feature_org=train_data_feature
y_train_org=y_train

""" Resampling data: Boost Straping """
data_resample = True
if data_resample is True:
    print "===================== Resampling Data...=========================="
    train_data_feature,y_train=ids_fe.resampling_data_smote(train_data_feature,y_train)

"""Random indexing data containing all categories"""
data_num=train_data_feature.shape[0]
idx=np.random.permutation(train_data_feature.shape[0])
data_idx=idx[:data_num]
train_data_feature=train_data_feature[data_idx, :]
y_train=y_train[data_idx]

train_num=train_data_feature.shape[0]
test_num=test_data.shape[0]

"""preparing feature matrix"""
test_data_feature=ids_fe.feature_matrix(test_data)

"""feature normalization"""
print "Normalize feature matrix ..."
sc_train=MinMaxScaler()
train_data_feature_normal=sc_train.fit_transform(train_data_feature)
test_data_feature_normal=sc_train.transform(test_data_feature)

Xtrain_org=sc_train.transform(train_data_feature_org)

Xtrain=train_data_feature_normal
Xtest=test_data_feature_normal


"""feature selection"""
feature_selection=True
if feature_selection is True:
    print "========================= Feature selection ...===================="
    feature_num=20
    sl_train=SelectKBest(chi2,k=feature_num)
    Xtrain=sl_train.fit_transform(Xtrain,y_train)
    Xtest=sl_train.transform(Xtest)

ids_util.visualize_data(Xtrain,y_train)
""" Spliting validation data """
print "============= Spliting validation set from training set ... ============"
Xtrain,Xval,y_train,y_val=ids_util.spilt_validate(Xtrain,y_train)

train_num=Xtrain.shape[0]
val_num=Xval.shape[0]

train_method='dnn'

"""training SVM model"""
if train_method=='svm':
 clf=ids_lm.train_svm(Xtrain,y_train,Xval,y_val,train_learning_model=False)


"""Training neural network"""
if train_method == 'dnn':
    Xtrain=Xtrain.astype('float32')
    Xtest=Xtest.astype('float32')
    clf=ids_lm.train_dnn(Xtrain,y_train,Xval,y_val)


"""Training random forest"""
if train_method == 'rf':
    # train_num = Xtrain.shape[0]
    # val_num = Xval.shape[0]
    # print "training on Random Forest Classifier..."
    # print "Training Num : %d  Validation Num : %d" % (train_num, val_num)
    clf = RandomForestClassifier()
    clf.fit(Xtrain, y_train)
    # train_err = clf.score(Xtrain, y_train)
    # val_err = clf.score(Xval, y_val)
    # print "Training error rate : %f" % train_err
    # print "Validation error rate : %f" % val_err


"""Write out preidcted label"""
write_file=True
if write_file is True:
    print "Write out predict label..."
    out_file='predict.csv'
    y_test_predict=clf.predict(Xtest)
    # y_test_predict = np.argmax(y_test_predict, axis=1)
    ids_util.write_predict(y_test_predict, out_file)









