from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import pickle as pk
from sklearn.metrics import accuracy_score

def train_svm(Xtrain,y_train,Xval,y_val,train_learning_model=False):
    train_num=Xtrain.shape[0]
    val_num=Xval.shape[0]
    train_model_file = 'learning_model_svm_smote.pkl'
    if train_learning_model is False:
        print "training on SVM multi-classifier..."
        print "Training Num : %d  Validation Num : %d" % (train_num, val_num)
        clf = OneVsRestClassifier(SVC())
        clf.fit(Xtrain, y_train)

        print "Write out learning model ..."
        pk.dump(clf, open(train_model_file, 'w'))
    else:
        print "Load SVM model ..."
        clf = pk.load(open(train_model_file, 'r'))

    print "Accuracy Measure on training set and validation set"
    y_train_predict = clf.predict(Xtrain)
    y_val_predict = clf.predict(Xval)
    train_err = accuracy_score(y_train, y_train_predict)
    val_err = accuracy_score(y_val, y_val_predict)
    print "Training error rate : %f" % train_err
    print "Validation error rate : %f" % val_err

    return clf

def train_dnn(Xtrain,y_train,Xval,y_val):
    from keras.models import Sequential
    from keras.layers import Activation,Dense,Dropout
    from keras.optimizers import Adam
    from keras.utils import np_utils
    print "Training Model by DNN...."
    nb_classes=5
    y_train=np_utils.to_categorical(y_train.astype(int),nb_classes)
    y_val=np_utils.to_categorical(y_val.astype(int),nb_classes)
    model=Sequential()
    model.add(Dense(64,input_shape=(Xtrain.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    model.fit(Xtrain,y_train,batch_size=128,nb_epoch=50,validation_data=(Xval,y_val))
    return model