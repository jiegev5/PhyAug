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
lst = ["inverted_data_5_src_model_sd_2pct_noise",
       "inverted_data_10_src_model_sd_2pct_noise",
       "inverted_data_15_src_model_sd_2pct_noise",
       "inverted_data_25_src_model_sd_2pct_noise",
       "inverted_data_50_src_model_sd_2pct_noise",
       "inverted_data_75_src_model_sd_2pct_noise",
       "inverted_data_100_src_model_sd_2pct_noise",
       "inverted_data_125_src_model_sd_2pct_noise",
       "inverted_data_150_src_model_sd_2pct_noise",
       "inverted_data_175_src_model_sd_2pct_noise",
       "inverted_data_200_src_model_sd_2pct_noise",
       "inverted_data_225_src_model_sd_2pct_noise",
       "inverted_data_250_src_model_sd_2pct_noise",
       "inverted_data_275_src_model_sd_2pct_noise",
       "inverted_data_300_src_model_sd_2pct_noise",
       "inverted_data_325_src_model_sd_2pct_noise",
       "inverted_data_350_src_model_sd_2pct_noise",
       "inverted_data_375_src_model_sd_2pct_noise",
       "inverted_data_400_src_model_sd_2pct_noise",
       "inverted_data_425_src_model_sd_2pct_noise",
       "inverted_data_500_src_model_sd_2pct_noise",
       "inverted_data_600_src_model_sd_2pct_noise",
       "inverted_data_700_src_model_sd_2pct_noise",
       "inverted_data_800_src_model_sd_2pct_noise",
       "inverted_data_900_src_model_sd_2pct_noise",
       "inverted_data_1000_src_model_sd_2pct_noise",
       "inverted_data_2000_src_model_sd_2pct_noise",
       "inverted_data_3000_src_model_sd_2pct_noise",
       "inverted_data_4000_src_model_sd_2pct_noise",
       "inverted_data_5000_src_model_sd_2pct_noise",
       "inverted_data_6000_src_model_sd_2pct_noise",
       "inverted_data_7000_src_model_sd_2pct_noise",
       "inverted_data_8000_src_model_sd_2pct_noise",
       "inverted_data_9000_src_model_sd_2pct_noise",
       "inverted_data_10000_src_model_sd_2pct_noise",
       "inverted_data_11000_src_model_sd_2pct_noise",
       "inverted_data_12000_src_model_sd_2pct_noise",
       "inverted_data_13000_src_model_sd_2pct_noise",
       "inverted_data_14000_src_model_sd_2pct_noise",
       "inverted_data_15000_src_model_sd_2pct_noise",
       "inverted_data_16000_src_model_sd_2pct_noise",
       "inverted_data_17000_src_model_sd_2pct_noise",
       "inverted_data_18000_src_model_sd_2pct_noise",
       "inverted_data_19000_src_model_sd_2pct_noise",
       "inverted_data_20000_src_model_sd_2pct_noise",
       "inverted_data_21000_src_model_sd_2pct_noise",
       "inverted_data_25000_src_model_sd_2pct_noise"
      ]


perturb = [2]
output_dir = 'perturbation_result/normalized_class_400'
dim = 20
batch_size = 100
GPU_NO = 0
n_label = dim*dim

model = 'svm'

if model == "mlp":
    filename = 'mlp_model_accuracy_on_aug_data_eval_2pct_perturb_class_400_shmoo.csv'
if model == "svm":
    filename = 'svm_model_accuracy_on_aug_data_eval_2pct_perturb_class_400_shmoo.csv'
# option is 'w'
with open(os.path.join(output_dir,filename), 'w', newline = '') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow(['No Scr','Test Accuracy'])
f.close()

for p in perturb:
    folder = "data/inverted_data_sd_"+str(p)+"pct_normalized_noise"
    train_dir = [os.path.join(folder,f) for f in lst]
    test_dir = os.path.join(folder,'true_model_data')
    for train_path in train_dir:
        file = [f for f in os.listdir(train_path) if re.match(r'model_.*_src_.*_data.*\.csv', f)]
        train_ = shuffle(np.concatenate([read_csv_to_np_with_dim(train_path, f, dim) for f in file]))
        files = [f for f in os.listdir(test_dir) if re.match(r'model_.*_src_.*_data.*\.csv', f)]
        test_ = shuffle(np.concatenate([read_csv_to_np_with_dim(test_dir, f, dim) for f in files]))
        N = 25000 if len(train_) > 30000 else len(train_)
        start_time = time.time()
        if model == 'mlp':
            acc = mlp_model(train_[:N,:],test_[-4000:,:],batch_size,n_label,GPU_NO)
        if model == 'svm':
            model_infor = svm_model_for_invert(train_[:N,:],train_[-4000:,:])
            acc = model_infor[1]
        print("time taken to complete 1 set training: ", time.time()-start_time)

        with open(os.path.join(output_dir,filename), 'a', newline = '') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([opts,acc])
        f.close()
    ###################################################################################################
    ###################################################################################################
