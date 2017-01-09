import numpy as np
import pandas as pd
import IDS_utility as ids_util
import pickle as pk
from imblearn.over_sampling import SMOTE


def resampling_data_smote(train_data,y_train):
    class_size = ids_util.get_class_size(y_train)
    class_data_num_avg = int(np.mean(class_size))

    # drop data
    majority_idx = np.nonzero(class_size >= class_data_num_avg)[0]
    for i in majority_idx:
        class_data_num = class_size[i]
        class_data_idx = np.nonzero(y_train == i)[0]
        delete_idx = np.random.permutation(class_data_num)[:class_data_num-class_data_num_avg]
        delete_idx = class_data_idx[delete_idx]
        train_data = np.delete(train_data, delete_idx, axis=0)
        y_train = np.delete(y_train, delete_idx, axis=0)

    # resmpling data
    for i in range(len(class_size)-len(majority_idx)):
        sm = SMOTE()
        train_data, y_train = sm.fit_sample(train_data, y_train)

    return train_data, y_train

def resampling_data(train_data,y_train):
    class_size = ids_util.get_class_size(y_train)
    class_data_num_avg = int(np.mean(class_size))
    idx_label = np.array([], dtype=int)
    train_data_resample=np.empty((0, train_data.shape[1]))

    for i in range(5):
        class_data_num = class_size[i]
        class_data_idx = np.nonzero(y_train == i)[0]
        if class_data_num < class_data_num_avg:
            # resampling data
            resample_idx = np.random.choice(class_data_num, class_data_num_avg, replace=True)
            data_idx = class_data_idx[resample_idx]
            class_data = train_data[data_idx,:]
        elif class_data_num > class_data_num_avg:
            # drop extra data
            sample_idx = np.random.permutation(class_data_num)[:class_data_num_avg]
            data_idx = class_data_idx[sample_idx]
            class_data = train_data[data_idx,:]
        else:
            data_idx = class_data_idx
            class_data = train_data[data_idx,:]

        train_data_resample=np.append(train_data_resample,class_data,axis=0)
        idx_label = np.append(idx_label, data_idx)

    y_train = y_train[idx_label]

    return train_data_resample,y_train


def feature_matrix(train_data):
    le_train_file = 'le_train.pkl'
    enc_train_file = 'enc_train.pkl'

    train_num=train_data.shape[0]
    cat_feature_idx = [1, 2, 3]
    cat_feature_train = np.empty((train_num, 0), dtype=int)

    le_train_list = pk.load(open(le_train_file, 'r'))

    for i in range(len(le_train_list)):
        idx = cat_feature_idx[i]
        le_train = le_train_list[i]
        # transform training feature
        cat_feature_train_list = train_data[idx].tolist()
        cat_feature_train_list = ids_util.check_list(cat_feature_train_list, le_train.classes_)
        cat_feature_train_label = le_train.transform(cat_feature_train_list).reshape((train_num, 1))
        cat_feature_train = np.append(cat_feature_train, cat_feature_train_label, axis=1)

    enc_train = pk.load(open(enc_train_file, 'r'))

    cat_feature_train = enc_train.transform(cat_feature_train).toarray()

    train_data = train_data.drop(train_data.columns[cat_feature_idx], axis=1)
    train_data = train_data.as_matrix()
    train_data_feature = np.append(train_data, cat_feature_train, axis=1)

    return train_data_feature


