import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt
import IDS_feature_extract as ids_fe
import pickle as pk

class_label_set = ['normal', 'dos', 'u2r', 'r2l', 'probe']


def write_predict(Y_predict, outputFilename):
    idxList=np.arange(Y_predict.shape[0])+1
    d={'id':pd.Series(idxList),'label':pd.Series(Y_predict.flatten())}
    predict=pd.DataFrame(d)
    predict.to_csv(outputFilename,index=False)


def check_list(input_list,check_ele):
    input_list_modify=[]
    for ele in input_list:
        if ele not in check_ele:
            input_list_modify.append('other')
        else:
            input_list_modify.append(ele)
    return input_list_modify


def get_class_size(y_train):
    class_num=np.zeros(5,dtype=int)
    for i in range(5):
        class_num[i]=np.sum(y_train==i)
    return class_num


def load_training(train_file,type_file,train_le=True):
    train_data = pd.read_csv(train_file, delimiter=",", header=None)
    type_data = pd.read_csv(type_file, delimiter=" ", header=None)
    train_num=train_data.shape[0]

    if train_le is False:
        train_label_encoder(train_data)

    """processing training label"""
    label_idx = 41
    cat_label_list = train_data[label_idx].tolist()
    cat_label_list = [w.strip('.') for w in cat_label_list]

    # convert to 5 types
    type_list = type_data[0].tolist()
    type_convert = type_data[1].tolist()
    cat_label = np.zeros((train_num,), dtype=int)
    for i in range(train_num):
        type_name = cat_label_list[i]
        if type_name != 'normal':
            type_idx = type_list.index(type_name)
            cat_label_list[i] = type_convert[type_idx]
        cat_label[i] = class_label_set.index(cat_label_list[i])
    y_train = cat_label
    train_data_feature=ids_fe.feature_matrix(train_data.iloc[:, :41])

    return train_data_feature,y_train


def spilt_validate(Xtrain,y_train):
    train_num=Xtrain.shape[0]
    val_num = int(train_num * 0.25)
    idx = np.random.permutation(train_num)
    Xtrain=Xtrain[idx,:]
    y_train=y_train[idx]

    Xval = Xtrain[:val_num, :]
    Xtrain = Xtrain[val_num:, :]
    y_val = y_train[:val_num]
    y_train = y_train[val_num:]

    return Xtrain,Xval,y_train,y_val


def visualize_data(Xtrain,y_train):
    data_num=10000
    Xtrain=Xtrain[0:data_num,:20]
    y_train=y_train[0:data_num]
    pca=PCA(n_components=2)
    X_vis=pca.fit_transform(Xtrain)
    color_list=['r', 'g','b', 'm', 'k']

    plt.figure()
    plt.hold(True)
    lp=[]
    for i in range(len(class_label_set)):
        vis_idx=y_train==i
        lp.append(plt.scatter(X_vis[vis_idx, 0], X_vis[vis_idx, 1],s=20, c=color_list[i], marker='o'))

    plt.legend((lp[0],lp[1],lp[2],lp[3],lp[4]),(class_label_set[0],class_label_set[1],class_label_set[2],class_label_set[3],class_label_set[4]))
    plt.show()


def train_label_encoder(train_data):
    # output module file name
    le_train_file = 'le_train.pkl'
    enc_train_file = 'enc_train.pkl'
    train_num = train_data.shape[0]
    cat_feature_idx = [1, 2, 3]
    cat_feature_train = np.empty((train_num, 0), dtype=int)

    le_train_list = []
    preserve_label_num = [3, 10, 5]
    ii = 0
    # train label encoder
    for idx in cat_feature_idx:
        # transform training feature
        cat_feature_train_list = train_data[idx].tolist()
        le_train = LabelEncoder()
        cat_feature_train_label = le_train.fit_transform(cat_feature_train_list).reshape((train_num, 1))

        # see each number of level
        class_num = []
        label_name_list = le_train.classes_
        for k in range(len(label_name_list)):
            class_num.append(0)
            for cat_feature in cat_feature_train_list:
                if cat_feature == label_name_list[k]:
                    class_num[k] += 1

        # preserve 5 maximum feature and one other
        class_num = np.array(class_num)

        if len(label_name_list) > preserve_label_num[ii]:
            idx_class = np.argsort(class_num)
            feature_idx = idx_class[-preserve_label_num[ii]:]
            preserve_label = label_name_list[feature_idx]
            cat_feature_train_list = check_list(cat_feature_train_list, preserve_label)
            le_train = LabelEncoder()
            cat_feature_train_label = le_train.fit_transform(cat_feature_train_list).reshape((train_num, 1))

        le_train_list.append(le_train)
        cat_feature_train = np.append(cat_feature_train, cat_feature_train_label, axis=1)
        ii +=1
    pk.dump(le_train_list, open(le_train_file, 'w'))

    # one hot encoder
    enc_train = OneHotEncoder()
    enc_train.fit_transform(cat_feature_train).toarray()
    pk.dump(enc_train, open(enc_train_file, 'w'))



