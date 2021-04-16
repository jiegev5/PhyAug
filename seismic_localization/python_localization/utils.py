import numpy as np
import csv
import re
import pylab as pl
import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()

    return 1-num_matches.float()/bs


def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('WRONG TENSOR SIZE')

def show_prob_cifar(p):


    p=p.data.squeeze().numpy()

    ft=15
    label = ('airplane', 'automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship','Truck' )
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()



def show_prob_mnist(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine')
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")


def show_prob_25_class(p):

    p=p.data.squeeze().numpy()

    ft=15
    # label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine')
    label = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")




def show_prob_fashion_mnist(p):


    p=p.data.squeeze().numpy()

    ft=15
    label = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Boot')
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")




import os.path
def check_mnist_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt')
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'mnist/train_data.pt')
        torch.save(train_label,path_data + 'mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'mnist/test_data.pt')
        torch.save(test_label,path_data + 'mnist/test_label.pt')
    return path_data

def check_fashion_mnist_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'fashion-mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'fashion-mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'fashion-mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'fashion-mnist/test_label.pt')
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('FASHION-MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'fashion-mnist/train_data.pt')
        torch.save(train_label,path_data + 'fashion-mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'fashion-mnist/test_data.pt')
        torch.save(test_label,path_data + 'fashion-mnist/test_label.pt')
    return path_data

def check_cifar_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'cifar/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'cifar/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'cifar/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'cifar/test_label.pt')
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('CIFAR dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=True,
                                        download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=False,
                                       download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(50000,3,32,32)
        train_label=torch.LongTensor(50000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0]
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'cifar/train_data.pt')
        torch.save(train_label,path_data + 'cifar/train_label.pt')
        test_data=torch.Tensor(10000,3,32,32)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0]
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'cifar/test_data.pt')
        torch.save(test_label,path_data + 'cifar/test_label.pt')
    return path_data

def svm_model(input_data):

    train, test = train_test_split(input_data, test_size=0.2)

    # print("Any missing sample in training set:",train.isnull().values.any())
    # print("Any missing sample in test set:",test.isnull().values.any(), "\n")

    # Seperating Predictors and Outcome values from train and test sets
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


    # visualize the scaled data with seaborn
    # fig, ax =plt.subplots(1,3)
    # sns.set_style('darkgrid')
    # sns.distplot(X_train[0],ax = ax[0])
    # sns.distplot(X_train_scaled[0], ax = ax[1])
    # sns.distplot(X_test_scaled[0], ax = ax[2])


    # ## Hyper parameter tuing using  grid search and cross validation

    #Libraries to Build Ensemble Model : Random Forest Classifier
    # Create the parameter grid based on the results of random search
    params_grid = [{'kernel': ['rbf'],
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'],
                    'C': [1, 10, 100, 1000]}]

    # Performing CV to tune parameters for best SVM fit
    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(X_train_scaled, Y_train.ravel())


    # View the accuracy score
    print('Best score for training data:', svm_model.best_score_,"\n")

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n")
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict(X_test_scaled)

    # Making the Confusion Matrix
    #print(pd.crosstab(Y_test_label, Y_pred_label, rownames=['Actual Activity'], colnames=['Predicted Activity']))
    print(confusion_matrix(Y_test,Y_pred))
    print("\n")
    # plot confusion matrix
    confusion_mat = confusion_matrix(Y_test,Y_pred)
    confusion_mat = torch.tensor(confusion_mat)
    show(1 - confusion_mat)
    # plt.close("all")
    print("\n")
    print(classification_report(Y_test,Y_pred))

    train_score = final_model.score(X_train_scaled , Y_train)
    test_score = final_model.score(X_test_scaled  , Y_test)
    print("Training set score for SVM: %f" % train_score)
    print("Testing  set score for SVM: %f" % test_score)

    svm_model.score

    # dump(final_model, filename.split('.')[0] + '.joblib')

    return len(input_data),test_score

def svm_model_for_invert(train,test):

    # train, test = train_test_split(input_data, test_size=0.2)

    # print("Any missing sample in training set:",train.isnull().values.any())
    # print("Any missing sample in test set:",test.isnull().values.any(), "\n")

    # Seperating Predictors and Outcome values from train and test sets
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


    # visualize the scaled data with seaborn
    # fig, ax =plt.subplots(1,3)
    # sns.set_style('darkgrid')
    # sns.distplot(X_train[0],ax = ax[0])
    # sns.distplot(X_train_scaled[0], ax = ax[1])
    # sns.distplot(X_test_scaled[0], ax = ax[2])


    # ## Hyper parameter tuing using  grid search and cross validation

    #Libraries to Build Ensemble Model : Random Forest Classifier
    # Create the parameter grid based on the results of random search
    params_grid = [{'kernel': ['rbf'],
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'],
                    'C': [1, 10, 100, 1000]}]

    # Performing CV to tune parameters for best SVM fit
    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(X_train_scaled, Y_train.ravel())


    # View the accuracy score
    print('Best score for training data:', svm_model.best_score_,"\n")

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n")
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict(X_test_scaled)

    # Making the Confusion Matrix
    #print(pd.crosstab(Y_test_label, Y_pred_label, rownames=['Actual Activity'], colnames=['Predicted Activity']))
    print(confusion_matrix(Y_test,Y_pred))
    print("\n")
    # plot confusion matrix
    confusion_mat = confusion_matrix(Y_test,Y_pred)
    confusion_mat = torch.tensor(confusion_mat)
    # show(1 - confusion_mat)
    # plt.close("all")
    print("\n")
    print(classification_report(Y_test,Y_pred))

    train_score = final_model.score(X_train_scaled , Y_train)
    test_score = final_model.score(X_test_scaled  , Y_test)
    print("Training set score for SVM: %f" % train_score)
    print("Testing  set score for SVM: %f" % test_score)

    svm_model.score

    # dump(final_model, filename.split('.')[0] + '.joblib')

    return len(train),test_score

def svm_model_for_invert_with_dim(train,test,dim):

    # train, test = train_test_split(input_data, test_size=0.2)

    # print("Any missing sample in training set:",train.isnull().values.any())
    # print("Any missing sample in test set:",test.isnull().values.any(), "\n")

    # Seperating Predictors and Outcome values from train and test sets
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


    # visualize the scaled data with seaborn
    # fig, ax =plt.subplots(1,3)
    # sns.set_style('darkgrid')
    # sns.distplot(X_train[0],ax = ax[0])
    # sns.distplot(X_train_scaled[0], ax = ax[1])
    # sns.distplot(X_test_scaled[0], ax = ax[2])


    # ## Hyper parameter tuing using  grid search and cross validation

    #Libraries to Build Ensemble Model : Random Forest Classifier
    # Create the parameter grid based on the results of random search
    params_grid = [# {'kernel': ['rbf'],
                   # 'gamma': [1e-3, 1e-4],
                   # 'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'],
                    'C': [1, 10, 100, 1000]}]

    # Performing CV to tune parameters for best SVM fit
    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(X_train_scaled, Y_train.ravel())


    # View the accuracy score
    print('Best score for training data:', svm_model.best_score_,"\n")

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n")
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict(X_test_scaled)

    # Making the Confusion Matrix
    #print(pd.crosstab(Y_test_label, Y_pred_label, rownames=['Actual Activity'], colnames=['Predicted Activity']))
    print(confusion_matrix(Y_test,Y_pred))
    print("\n")
    # plot confusion matrix
    confusion_mat = confusion_matrix(Y_test,Y_pred)
    confusion_mat = torch.tensor(confusion_mat)
    # show(1 - confusion_mat)
    # plt.close("all")
    print("\n")
    print(classification_report(Y_test,Y_pred))

    train_score = final_model.score(X_train_scaled , Y_train)
    test_score = final_model.score(X_test_scaled  , Y_test)
    print("Training set score for SVM: %f" % train_score)
    print("Testing  set score for SVM: %f" % test_score)

    svm_model.score
    y_true = Y_test.astype(int)
    y_pred = Y_pred.astype(int)
    err = get_err(y_true.reshape(-1,),y_pred,(dim,dim))
    # dump(final_model, filename.split('.')[0] + '.joblib')

    return len(train),test_score,np.mean(err)

# how many classes per side, for example: if 25 classes, then each dim is 5 for x and y
Class_dim = 10

def read_csv_to_np(dir_, filename):
    ##read csv file
    # filename = 'model_ch_src_1000_rec_10_data_1.csv'
    # file_path = os.path.join(os.getcwd(),dir_)
    f = dir_ + '/' + filename
    raw_data = shuffle(pd.read_csv(f))
    print("granurity: ",Class_dim)

    # split into features and classes
    df1 = raw_data.iloc[:,:8]
    df2 = raw_data.iloc[:,8:]

    # process df2, convert loc into classes
    # src_loc = ((np.floor(df2) - 1).astype(int)).to_numpy()
    src_loc = np.floor((df2 - 1)/(100/Class_dim)).astype(int).to_numpy()
    # get the receiver loc
    src_linear_loc = np.array([np.ravel_multi_index(item, dims=(Class_dim,Class_dim), order='F') for item in src_loc]).reshape(1000,1)

    # map travel time from each sensor to the map
    feature = df1.to_numpy()

    final_out = np.concatenate((feature,src_linear_loc),axis = 1)

    return final_out

def read_csv_to_np_with_dim(dir_, filename, dim):
    ##read csv file
    f = os.path.join(dir_,filename)
    raw_data = shuffle(pd.read_csv(f))
    # print("granurity: ",dim)

    # split into features and classes
    df1 = raw_data.iloc[:,:8]
    df2 = raw_data.iloc[:,8:]

    # src_loc = ((np.floor(df2) - 1).astype(int)).to_numpy()
    src_loc = np.floor((df2 - 1)/(100/dim)).astype(int).to_numpy()
    # get the receiver loc
    src_linear_loc = np.array([np.ravel_multi_index(item, dims=(dim,dim), order='F') for item in src_loc]).reshape(1000,1)

    # map travel time from each sensor to the map
    feature = df1.to_numpy()

    final_out = np.concatenate((feature,src_linear_loc),axis = 1)

    return final_out

# https://github.com/iam-mhaseeb/Multi-Layer-Perceptron-MNIST-with-PyTorch/blob/master/mnist_mlp_exercise.ipynb
class four_layer_net(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3,  output_size):
        super(four_layer_net , self).__init__()

        self.layer1 = nn.Linear(input_size,hidden_size1,bias = True)
        self.layer2 = nn.Linear(hidden_size1,hidden_size2,bias = True)
        self.layer3 = nn.Linear(hidden_size2,hidden_size3,bias = True)
        self.layer4 = nn.Linear(hidden_size3,output_size,bias = True)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):

        y       = self.layer1(x) # COMPLETE HERE
        y_hat   = F.relu(y) # COMPLETE HERE
        self.dropout
        z       = self.layer2(y_hat) # COMPLETE HERE
        z_hat   = F.relu(z) # COMPLETE HERE
        self.dropout
        za      = self.layer3(z_hat) # COMPLETE HERE
        za_hat  = F.relu(za) # COMPLETE HERE
        self.dropout
        score   = self.layer4(za_hat)

        return score

class five_layer_net(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3,  hidden_size4, output_size):
        super(five_layer_net , self).__init__()

        self.layer1 = nn.Linear(input_size,hidden_size1,bias = True)
        self.layer2 = nn.Linear(hidden_size1,hidden_size2,bias = True)
        self.layer3 = nn.Linear(hidden_size2,hidden_size3,bias = True)
        self.layer4 = nn.Linear(hidden_size3,hidden_size4,bias = True)
        self.layer5 = nn.Linear(hidden_size4,output_size,bias = True)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):

        y       = self.layer1(x) # COMPLETE HERE
        y_hat   = F.relu(y) # COMPLETE HERE
        self.dropout
        z       = self.layer2(y_hat) # COMPLETE HERE
        z_hat   = F.relu(z) # COMPLETE HERE
        self.dropout
        za      = self.layer3(z_hat) # COMPLETE HERE
        za_hat  = F.relu(za) # COMPLETE HERE
        self.dropout
        zb      = self.layer4(za_hat) # COMPLETE HERE
        zb_hat  = F.relu(zb) # COMPLETE HERE
        self.dropout
        score   = self.layer5(zb_hat)

        return score

def from_1d_to_2d_coord(x,dim):
    '''conver a 1d index array to 2d index array, dim is a tuple with x,y dim'''
    index = np.unravel_index(x,dim,order='F')
    return np.array((index[0],index[1])).T

def obtain_dist(x,y):
    '''obtain pairwise euclidean distance of array x, y'''
    dist = []
    for i in range(len(x)):
        d = np.linalg.norm(x[i]-y[i])
        dist.append(d)
    return np.asarray(dist)

def get_err(y_true, y_pred, dim):
    # y_true: 1d array index
    # y_pred: 1d array index
    # dim: tuple with (shap_of_x,shape_of_y)
    unit_length = (100/dim[0])*10 # unit is meter
    y_true_2d = from_1d_to_2d_coord(y_true,dim)*unit_length
    y_pred_2d = from_1d_to_2d_coord(y_pred,dim)*unit_length
    return obtain_dist(y_true_2d,y_pred_2d)
'''
class MLP_4_layer(nn.Module):
    def __init__(self, num_features, dropout=0.25, n_hid=1024, n_label):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid//2, n_hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, n_label),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
        '''
