#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:58:07 2017

@author: xuanwei
"""
import numpy as np
from datetime import datetime
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
start=datetime.now()
file_train="/home/xuanwei/job/learnCNN/数字识别/train.csv"

X=np.array(pd.read_csv(file_train),dtype="float32")[0:,1:]
labels=np.array(pd.read_csv(file_train),dtype="uint8")[:,0]
labels=np_utils.to_categorical(labels)

model=Sequential()
model.add(Dense(800,init='normal',input_shape=(784,)))
model.add(Activation('sigmoid'))

model.add(Dense(800,init='normal'))
model.add(Activation('sigmoid'))

model.add(Dense(10,init='normal'))
model.add(Activation('softmax'))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=False)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X,labels,batch_size=1000,nb_epoch=10,shuffle=True,verbose=1,validation_split=0.2)


file_test="/home/xuanwei/job/learnCNN/数字识别/test.csv"
X_test=np.array(pd.read_csv(file_test),dtype="float32")
#X_test=X_test.reshape(-1,28,28,1)
labels_test=model.predict_classes(X_test, batch_size=32, verbose=1)

#Y=labels_test.argmax(axis=1)
tableHeader=np.array(['ImageId','Label'])

res_index=np.arange(1,len(X_test)+1)
labels_test=np.column_stack((res_index,labels_test))
labels_test=np.row_stack((tableHeader,labels_test))
np.savetxt('/home/xuanwei/job/learnCNN/数字识别/fullConnect/test_res.csv', labels_test, fmt='%s',delimiter = ',') 
end=datetime.now()
print (end-start)
