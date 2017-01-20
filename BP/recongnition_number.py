# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import numpy.random as rd
from datetime import datetime
import pandas as pd
start=datetime.now()
file_train="/home/xuanwei/job/learnCNN/数字识别/train.csv"
lable=np.zeros([42000,10])
X=np.zeros([42000,785])
X=np.array(pd.read_csv(file_train))[0:,1:]
X_bias=np.ones(42000)
X=np.column_stack((X,X_bias))

lable_index=np.array(pd.read_csv(file_train)).T[0].T
lable_temp=np.arange(42000)
lable[lable_temp,lable_index]=1
#分训练集与验证集
X_train=X[:40000]
X_validation=X[40000:]
label_train=lable[:40000]
label_validation=lable[40000:]
#归一化
X_train=X_train/255
X_validation=X_validation/255
#初识化参数
W=rd.uniform(-0.3, 0.3, (1000,785))
V=rd.uniform(-0.3, 0.3, (10,1001))
mu=0.01
momentumV=0
momentumW=0
alpha=0.9
lambda_v=0.01
lambda_w=0.01
#训练集
iterations=100000
n=0
for j in range(iterations):
    # 验证集的损失
    if n>=len(X_train):
        cost_validation=0.0
        Z_hide=np.dot(X_validation,W.T)
                      
        A_hide=1/(1+np.exp((-1)*Z_hide))
        bias=np.ones(len(X_validation))
        A_hide=np.column_stack((A_hide,bias.reshape(-1,1)))
        Z_out=np.dot(A_hide,V.T)
        
        SUM=np.exp(Z_out).sum(axis=1)
        A_out=np.exp(Z_out)/(SUM.reshape(-1,1))
        cost_validation=-(label_validation*np.log(A_out)).sum()/(len(label_validation))
        predict_validation=(A_out.argmax(axis=1)==label_validation.argmax(axis=1)).mean()
        
        print (cost_validation)
        X_train_labels=np.column_stack((X_train,label_train))
        np.random.shuffle(X_train_labels)
        X_train= X_train_labels[:,:785]
        label_train=X_train_labels[:,785:]
        n=0
    X_train_temp=X_train[n:n+32]
    label_train_temp=label_train[n:n+32]
    Z_hide=np.dot(X_train_temp,W.T)
                  
    A_hide=1/(1+np.exp((-1)*Z_hide))
    bias=np.ones(len(X_train_temp))
    A_hide=np.column_stack((A_hide,bias.reshape(-1,1)))
    Z_out=np.dot(A_hide,V.T)
    
    
    SUM=np.exp(Z_out).sum(axis=1)
    A_out=np.exp(Z_out)/(SUM.reshape(-1,1))
    deta_A_out_Z_out=A_out-label_train_temp
    
    deta_Z_out_V=A_hide
    
    V_derivation=np.dot(deta_A_out_Z_out.T,deta_Z_out_V)/len(X_train_temp)+lambda_v*V/len(X_train_temp)
    
    
    A_hide= np.delete( A_hide,-1,axis=1)
    
    deta_A_hide=np.dot(deta_A_out_Z_out,V)
    deta_A_hide=np.delete(deta_A_hide,-1,axis=1)
    
    deta_A_hide_Z_hide= A_hide*(1- A_hide)
    
    deta_Z_hide_W=X_train_temp
    
    
    W_derivation=np.dot((deta_A_hide*deta_A_hide_Z_hide).T,deta_Z_hide_W)/len(X_train_temp)+lambda_w*W/len(X_train_temp)

    momentumV=alpha*momentumV-mu*V_derivation
    momentumW=alpha*momentumW-mu*W_derivation
    V=V+momentumV
    W=W+momentumW
    n=n+32
print (predict_validation)
np.savetxt('/home/xuanwei/job/learnCNN/数字识别/W.csv',W,delimiter=',')
np.savetxt('/home/xuanwei/job/learnCNN/数字识别/V.csv',V,delimiter=',')     
 
#测试集  
file_test="/home/xuanwei/job/learnCNN/数字识别/test.csv"
W=np.array(pd.read_csv('/home/xuanwei/job/learnCNN/数字识别/W.csv',header=None))
V=np.array(pd.read_csv('/home/xuanwei/job/learnCNN/数字识别/V.csv',header=None))
X_test=np.array(pd.read_csv(file_test)) 
 
X_bias=np.ones(len(X_test))
X_test=np.column_stack((X_test,X_bias)) 
X_test=X_test/255
Z_hide=np.dot(X_test,W.T)
              
A_hide=1/(1+np.exp((-1)*Z_hide))
bias=np.ones(len(X_test))
A_hide=np.column_stack((A_hide,bias.reshape(-1,1)))
Z_out=np.dot(A_hide,V.T)


SUM=np.exp(Z_out).sum(axis=1)
A_out=np.exp(Z_out)/SUM.reshape(-1,1)
Y=A_out.argmax(axis=1)
tableHeader=np.array(['ImageId','Label'])

res_index=np.arange(1,len(X_test)+1)
Y=np.column_stack((res_index,Y))
Y=np.row_stack((tableHeader,Y))
np.savetxt('/home/xuanwei/job/learnCNN/数字识别/test_res.csv', Y, fmt='%s',delimiter = ',') 
end=datetime.now()
print (end-start)

