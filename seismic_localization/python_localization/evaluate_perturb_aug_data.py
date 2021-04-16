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
lst = ["inverted_data_50_src_model_sd_",
       "inverted_data_100_src_model_sd_",
       "inverted_data_200_src_model_sd_",
       "inverted_data_300_src_model_sd_",
       "inverted_data_400_src_model_sd_",
       "inverted_data_500_src_model_sd_",
       "inverted_data_600_src_model_sd_",
       "inverted_data_700_src_model_sd_",
       "inverted_data_800_src_model_sd_",
       "inverted_data_900_src_model_sd_",
       "inverted_data_1000_src_model_sd_",
       "inverted_data_2000_src_model_sd_",
       "inverted_data_3000_src_model_sd_",
       "inverted_data_4000_src_model_sd_",
       "inverted_data_5000_src_model_sd_",
       "inverted_data_6000_src_model_sd_",
       "inverted_data_7000_src_model_sd_",
       "inverted_data_8000_src_model_sd_"
      ]
def gen_file(f,n):
    return f+str(n)+"pct_noise"

perturb = [1,2,3,4,5,6,7,8,9,10]
output_dir = 'perturbation_result/normalized_class_400'
dim = 20
batch_size = 100
GPU_NO = 0
n_label = dim*dim

model = 'svm'

if model == "mlp":
    filename = 'mlp_model_accuracy_on_aug_data_eval_perturb_class_400.csv'
if model == "svm":
    filename = 'svm_model_accuracy_on_aug_data_eval_perturb_class_400.csv'
# option is 'w'
with open(os.path.join(output_dir,filename), 'w', newline = '') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow(['p','No Scr','Test Accuracy'])
f.close()

for p in perturb:
    folder = "data/inverted_data_sd_"+str(p)+"pct_normalized_noise"
    new_lst = [gen_file(f,p) for f in lst]
    train_dir = [os.path.join(folder,f) for f in new_lst]
    test_dir = os.path.join(folder,'true_model_data')
    for train_path in train_dir:
        file = [f for f in os.listdir(train_path) if re.match(r'model_.*_src_.*_data.*\.csv', f)]
        train = shuffle(np.concatenate([read_csv_to_np_with_dim(train_path, f, dim) for f in file]))
        files = [f for f in os.listdir(test_dir) if re.match(r'model_.*_src_.*_data.*\.csv', f)]
        test = shuffle(np.concatenate([read_csv_to_np_with_dim(test_dir, f, dim) for f in files]))
        N = 25000 if len(train) > 30000 else len(train)
        start_time = time.time()
        if model == 'mlp':
            acc = mlp_model(train[:N,:],test[-4000:,:],batch_size,n_label,GPU_NO)
        if model == 'svm':
            model_infor = svm_model_for_invert(train[:N,:],test[-4000:,:])
            acc = model_infor[1]
        print("time taken to complete 1 set training: ", time.time()-start_time)

        with open(os.path.join(output_dir,filename), 'a', newline = '') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([p,opts,acc])
        f.close()
    ###################################################################################################
    ###################################################################################################
