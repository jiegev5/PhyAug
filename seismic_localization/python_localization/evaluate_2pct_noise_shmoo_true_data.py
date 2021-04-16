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
from utils import *
from mlp_model import mlp_model
import time


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

###################################################################################################
###################################################################################################
perturb = [2]
output_dir = 'perturbation_result/normalized_class_400'
dim = 20
batch_size = 100
GPU_NO = 0
n_label = dim*dim

model = 'svm'

if model == "mlp":
    filename = 'mlp_model_accuracy_on_true_data_eval_2pct_perturb_class_400_shmoo.csv'
if model == "svm":
    filename = 'svm_model_accuracy_on_true_data_eval_2pct_perturb_class_400_shmoo.csv'
# option is 'w'
with open(os.path.join(output_dir,filename), 'w', newline = '') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow(['No Scr','Test Accuracy'])
f.close()

for p in perturb:
    train_path = "data/inverted_data_sd_" + str(p) + "pct_normalized_noise/true_model_data"
    option = [400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000,9200,9400,9600,9800,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,25000]

    for opts in option:
        file = [f for f in os.listdir(train_path) if re.match(r'model_.*_src_.*_data.*\.csv', f)]
        train_ = shuffle(np.concatenate([read_csv_to_np_with_dim(train_path, f, dim) for f in file]))

        start_time = time.time()
        if model == 'mlp':
            acc = mlp_model(train_[:opts,:],train_[-4000:,:],batch_size,n_label,GPU_NO)
        if model == 'svm':
            model_infor = svm_model_for_invert(train_[:opts,:],train_[-4000:,:])
            acc = model_infor[1]
        print("time taken to complete 1 set training: ", time.time()-start_time)

        with open(os.path.join(output_dir,filename), 'a', newline = '') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([opts,acc])
        f.close()
    ###################################################################################################
    ###################################################################################################
