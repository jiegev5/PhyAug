import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)


## dataloader for spectrogram
def load_spectrogram(fpath):
    images = np.array([])
    labels = np.array([])
    flist = glob.glob(os.path.join(fpath,"*.csv"))
    # index = 0
    for f in flist:
        data = np.genfromtxt(f, delimiter=',')
        samples = data.shape[0]
        images = np.vstack([images,data.reshape(-1,5,32)]) if images.size else data.reshape(-1,5,32)
        
        index = f.split('_')[1]
        label = np.zeros(samples) + int(index)
        labels = np.concatenate((labels,label)) if labels.shape else label
        # print(f'file: {f}, index: {index}, samples: {samples}')
        # index += 1
    # normalize
    # mean = images.mean()
    # std = images.std()
    # images = (images - mean) / std  # normalize data

    return images.reshape(-1,1,5,32), labels

def load_spectrogram_train(fpath,samples_per_class):
    images = []
    labels = []
    num_classes = 4
    with open(fpath, 'rb') as rfile:
        train_dataset =  pickle.load(rfile)
    # for image in train_dataset['features']:
    #     # print(image.min(),image.max())
    #     images.append((image/255)-.5)
    # for label in train_dataset['labels']:
    #     # labels.append(np.eye(num_classes)[label])
    #     labels.append(label)

    images = np.array(train_dataset['features'])
    labels = np.array(train_dataset['labels'])

    # sample N data from each class
    N = samples_per_class
    unique, counts = np.unique(labels, return_counts=True)
    new_images = np.array([])
    new_labels = np.array([])
    start = 0
    for i in range(len(unique)):
        # random list
        list_ = np.random.random_integers(counts[i]-1,size=(N)) + start
        temp_m = images[list_]
        temp_l = labels[start:start+N]
        new_images = np.vstack([new_images,temp_m]) if new_images.size else temp_m
        new_labels = np.concatenate((new_labels,temp_l)) if new_labels.size else temp_l
        start = start + counts[i]
    print(new_images.shape,new_labels.shape)


    return new_images.reshape(-1,1,5,32), new_labels

class readSPEC:
    def __init__(self,data_root,mode):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data
        # opt = '' # 
        # opt = '_4_classes'
        # opt = '_exp14'
        data_root = data_root
        train_dir = os.path.join(data_root,mode)
        # for evaluating the effect of class samples
        # img, lab = load_spectrogram_train(train_dir,10) 
        img, lab = load_spectrogram(train_dir)
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels)    

        # load test data
        test_dir = os.path.join(data_root,mode)
        img, lab = load_spectrogram(test_dir)
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels)   

        # load validation data

        # img, lab = load_batch('./data/traffic-signs-data/valid.p')
        # self.validation_data.extend(img)
        # self.validation_labels.extend(lab)

        # self.validation_data = np.array(self.validation_data,dtype=np.float32)
        # self.validation_labels = np.array(self.validation_labels)  

class SPEC(Dataset):
    def __init__(self,dataroot,mode):
        self.data = torch.from_numpy(getattr(readSPEC(dataroot,mode),'{}_data'.format(mode))).float() # .permute(0,3,1,2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readSPEC(dataroot,mode),'{}_labels'.format(mode))).long()
        # print(self.target[0])
        
        # normalize data
        # self.transform = transforms.Normalize((0.5,), (0.5,))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
        # return x, y
    
    def __len__(self):
        return len(self.data)

def load_spec(dataroot='./data',batch_size=32):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(SPEC(dataroot,'train'),
    batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(SPEC(dataroot,'test'),
    batch_size=batch_size, shuffle=False, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(GTSRB('val'),
    # batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        print(X[i, 0], X[i, 1], str(y[i]))
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 4.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            # imagebox = offsetbox.AnnotationBbox(
            #     offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            #     X[i])
            # ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
