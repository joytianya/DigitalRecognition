#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:56:37 2017

@author: xuanwei
"""

import numpy as np
from datetime import datetime
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten

from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


start=datetime.now()
file_train="/home/xuanwei/job/learnCNN/数字识别/train.csv"
#label=np.zeros([42000,10])

X=np.array(pd.read_csv(file_train),dtype="float32")[0:,1:]
X=X.reshape(-1,28,28,1)

label=np.array(pd.read_csv(file_train),dtype="uint8")[:,0]
label=np_utils.to_categorical(label,10)

model=Sequential()
model.add(Convolution2D(4,3,3,border_mode='same',input_shape=(28, 28, 1)),)
model.add(Activation('tanh'))


model.add(Convolution2D(8,3,3,border_mode='same'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(16,3,3,border_mode='same'))
model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3,border_mode='same'))
model.add(Activation('tanh'))

model.add(Flatten())
model.add(Dense(128,init='normal'))
model.add(Activation('tanh'))

model.add(Dense(10,init='normal'))
model.add(Activation('softmax'))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(X,label,batch_size=100,nb_epoch=100,shuffle=True,verbose=1,validation_split=0.2)
#testSet
file_test="/home/xuanwei/job/learnCNN/数字识别/test.csv"
X_test=np.array(pd.read_csv(file_test),dtype="float32")
X_test=X_test.reshape(-1,28,28,1)
labels_test=model.predict_classes(X_test, batch_size=32, verbose=1)

#Y=labels_test.argmax(axis=1)
tableHeader=np.array(['ImageId','Label'])

res_index=np.arange(1,len(X_test)+1)
labels_test=np.column_stack((res_index,labels_test))
labels_test=np.row_stack((tableHeader,labels_test))
np.savetxt('/home/xuanwei/job/learnCNN/数字识别/CNN/test_res.csv', labels_test, fmt='%s',delimiter = ',') 
end=datetime.now()
print (end-start)

