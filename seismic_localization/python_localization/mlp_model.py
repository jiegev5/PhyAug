import os
import numpy as np
import csv
import re
import pylab as pl
import pandas as pd
import utils
from joblib import dump, load
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
from utils import read_csv_to_np,read_csv_to_np_with_dim,four_layer_net,five_layer_net
import time

def mlp_model(train,test,batch_size,n_label,GPU_NO):
    X_train = train[:,:8]
    Y_train = train[:,8:]
    X_test = test[:,:8]
    Y_test = test[:,8:]

    # Dimension of Train and Test set
    print("Dimension of Train set",X_train.shape)
    print("Dimension of Test set",X_test.shape,"\n")

    # Scaling the Train and Test feature set

    scaler = StandardScaler()
    # ?? why use fit_transform for trin but transorm for test??
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # switch between CPU and GPU
    device= torch.device("cuda")
    torch.cuda.set_device(GPU_NO)
    # device= torch.device("cpu")

    # train_data,train_label,test_data,test_label = map(torch.tensor,(X_train_scaled,Y_train,X_test_scaled,Y_test))
    # why series data cannot directly convert to tensor
    train_data = torch.tensor(X_train_scaled).float()
    train_label = torch.tensor(np.array(Y_train)).long()
    test_data = torch.tensor(X_test_scaled).float()
    test_label = torch.tensor(np.array(Y_test)).long()


    # reshape (R,1) to (R,)
    train_label = train_label.reshape(train_label.shape[0],)
    test_label = test_label.reshape(test_label.shape[0],)


    def eval_on_test_set():

        running_error=0
        num_batches=0


        for i in range(0,4000,bs):

            # extract the minibatch
            minibatch_data =  test_data[i:i+bs]
            minibatch_label= test_label[i:i+bs]

            # send them to the gpu
            minibatch_data=minibatch_data.to(device)
            minibatch_label=minibatch_label.to(device)

            # reshape the minibatch
            inputs = minibatch_data.view(bs,8)
            inputs = inputs.to(device)
            minibatch_label = minibatch_label.to(device)
            # feed it to the network
            scores=net( inputs )

            # compute the error made on this batch
            error = utils.get_error( scores , minibatch_label)

            # add it to the running error
            running_error += error.item()

            num_batches+=1


        # compute error rate on the full test set
        total_error = running_error/num_batches

        print( 'error rate on test set =', total_error*100 ,'percent')

    start = time.time()

    net=four_layer_net(8,1024,1024,512,n_label)
    # net = five_layer_net(8,512,1024,1024,512,100)
    net= net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD( net.parameters() , lr=0.01 )
    bs= batch_size
    lr = 0.5 # initial learning rate
    for epoch in range(150):
        # learning rate strategy : divide the learning rate by 1.5 every 10 epochs
        if epoch%10==0 and epoch>10:
            lr = lr / 1.5
        # create a new optimizer at the beginning of each epoch: give the current learning rate.
        optimizer=torch.optim.SGD( net.parameters() , lr=lr )
        running_loss=0
        running_error=0
        num_batches=0
        shuffled_indices=torch.randperm(len(train_label))
        for count in range(0,len(train_label),bs):
            # forward and backward pass
            optimizer.zero_grad()
            indices=shuffled_indices[count:count+bs]
            minibatch_data =  train_data[indices]
            minibatch_label= train_label[indices]
            inputs = minibatch_data.view(bs,8)
            inputs = inputs.to(device)
            minibatch_label = minibatch_label.to(device)
            inputs.requires_grad_()
            scores=net( inputs )
            loss =  criterion( scores , minibatch_label)
            loss.backward()
            optimizer.step()
            # compute some stats
            running_loss += loss.detach().item()
            error = utils.get_error( scores.detach() , minibatch_label)
            running_error += error.item()
            num_batches+=1
        # once the epoch is finished we divide the "running quantities"
        # by the number of batches
        total_loss = running_loss/num_batches
        total_error = running_error/num_batches
        elapsed_time = time.time() - start
        # every 10 epoch we display the stats
        # and compute the error rate on the test set
        if epoch % 10 == 0 :

            print(' ')
            print('epoch=',epoch, ' time=', elapsed_time,
                ' loss=', total_loss , ' error=', total_error*100 ,'percent lr=', lr)
            eval_on_test_set()
    test_data = test_data.cuda()
    scores = net( test_data )
    out = F.softmax(scores)
    preds = torch.argmax(out,dim=1)
    # load to cpu
    preds = preds.cpu()
    print(confusion_matrix(test_label,preds))
    print("\n")
    print(classification_report(test_label,preds))
    p_, r, f1, s = precision_recall_fscore_support(test_label,preds)
    return p_.mean()
    ###################################################################################################
    ###################################################################################################
