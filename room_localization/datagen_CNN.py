import torch
from torch._C import dtype
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import cm

def load_mnist(args):
    # torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_m/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                # batch_size=32, shuffle=False, **kwargs)
                batch_size=32, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=1, shuffle=True, **kwargs)
            # batch_size=32, shuffle=True, **kwargs)

    return train_loader, test_loader

def load_batch(fpath):
    images = []
    labels = []
    num_classes = 43
    with open(fpath, 'rb') as rfile:
        train_dataset =  pickle.load(rfile)
    for image in train_dataset['features']:
        # print(image.min(),image.max())
        images.append((image/255)-.5)
    for label in train_dataset['labels']:
        # labels.append(np.eye(num_classes)[label])
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

class readGTSRB:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data

        img, lab = load_batch('./data/traffic-signs-data/train.p')
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels)    

        # load test data

        img, lab = load_batch('./data/traffic-signs-data/test.p')
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels)   

        # load validation data

        img, lab = load_batch('./data/traffic-signs-data/valid.p')
        self.validation_data.extend(img)
        self.validation_labels.extend(lab)

        self.validation_data = np.array(self.validation_data,dtype=np.float32)
        self.validation_labels = np.array(self.validation_labels)  

class GTSRB(Dataset):
    def __init__(self,mode):
        self.data = torch.from_numpy(getattr(readGTSRB(),'{}_data'.format(mode))).float().permute(0,3,1,2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readGTSRB(),'{}_labels'.format(mode))).long()
        # print(self.target[0])
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
    
    def __len__(self):
        return len(self.data)

def load_gtsrb():
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(GTSRB('train'),
    batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(GTSRB('test'),
    batch_size=1, shuffle=False, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(GTSRB('val'),
    # batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader


### RIR data loader
class RIR(Dataset):
    def __init__(self,mode):
        self.data = torch.from_numpy(getattr(readRIR(),'{}_data'.format(mode))).float()##.view(-1,1,-1)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readRIR(),'{}_labels'.format(mode))).long()
        # print(self.target.shape)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
    
    def __len__(self):
        return len(self.data)

class readRIR:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data

        img, lab = load_RIR('data/RIR/train.npy')
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels)    

        # load test data

        img, lab = load_RIR('data/RIR/test.npy')
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels)   

        # load validation data

        img, lab = load_RIR('data/RIR/valid.npy')
        self.validation_data.extend(img)
        self.validation_labels.extend(lab)

        self.validation_data = np.array(self.validation_data,dtype=np.float32)
        self.validation_labels = np.array(self.validation_labels)  

def load_RIR(fpath):
    images = []
    labels = []
    num_classes = 10
    # with open(fpath, 'rb') as rfile:
    train_dataset =  np.load(fpath)
    images = train_dataset[:,:-1]
    # reshape to 3 channel
    x,y = images.shape
    print(x,y)
    images = images.reshape((x,1,y))
    labels = train_dataset[:,-1:]
    # print("shape is: ",images.shape,labels.shape)
    # for image in train_dataset[:-1]:
    #     # print(image.min(),image.max())
    #     print("image shape: ",image.shape)
    #     images.append(image)
    # for label in train_dataset[-1:]:
    #     # labels.append(np.eye(num_classes)[label])
    #     print("label shape: ",label.shape)
    #     labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def load_rir():
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(RIR('train'),
    batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(RIR('test'),
    batch_size=1, shuffle=False, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(RIR('val'),
    # batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader

## dataloader for spectrogram
def load_spectrogram(fpath):
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
    # N = 500
    # unique, counts = np.unique(labels)
    # new_images = np.array([])
    # new_labels = np.array([])
    # start = 0
    # for i in range(len(unique)):
    #     temp_m = images[start:start+N]
    #     temp_l = labels[start:start+N]
    #     new_images = np.vstack([new_images,temp_m]) if new_images.size else temp_m
    #     new_labels = np.vstack([new_labels,temp_l]) if new_labels.size else temp_l
    #     start = start + counts[i]
    # print(new_images.shape,new_labels.shape)


    # return images.reshape(-1,1,12,48), labels
    return images.reshape(-1,163), labels

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


    return new_images.reshape(-1,163), new_labels

class readSPEC:
    def __init__(self,opt,sample_per_class):
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
        data_root = '/data3/wenjie/echo_localization/DATA_FOLDER/cnn_data/spec'
        train_dir = os.path.join(data_root,'train_{}.p'.format(opt))
        # for evaluating the effect of class samples
        # img, lab = load_spectrogram_train(train_dir,10) 
        if sample_per_class is None:
            img, lab = load_spectrogram(train_dir)
        else:
            img, lab = load_spectrogram_train(train_dir,samples_per_class=sample_per_class)
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels,dtype=np.long)    

        # load test data
        test_dir = os.path.join(data_root,'test_{}.p'.format(opt))
        img, lab = load_spectrogram(test_dir)
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels,dtype=np.long)   

        # load validation data

        # img, lab = load_batch('./data/traffic-signs-data/valid.p')
        # self.validation_data.extend(img)
        # self.validation_labels.extend(lab)

        # self.validation_data = np.array(self.validation_data,dtype=np.float32)
        # self.validation_labels = np.array(self.validation_labels)  
s = 1
color_jitter = torchvision.transforms.ColorJitter(
    0.4 * s, 0.4 * s, 0.4 * s, 0.2 * s
)

train_transform = torchvision.transforms.Compose(
    [   
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomResizedCrop(size=(12,48)),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomApply([color_jitter], p=0.8),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        
    ]
)
test_transform = torchvision.transforms.Compose(
    [   
        torchvision.transforms.RandomResizedCrop(size=(12,48)),
        torchvision.transforms.ToTensor(),   
        torchvision.transforms.Normalize((0.5,), (0.5,)),
           
    ]
)    

class SPEC_WT(Dataset):
    '''
    with transform
    '''
    def __init__(self,mode,opt,sample_per_class=None):
        self.data = getattr(readSPEC(opt,sample_per_class),'{}_data'.format(mode)) # .permute(0,3,1,2)
        # print(self.data.shape)
        self.target = getattr(readSPEC(opt,sample_per_class),'{}_labels'.format(mode))
        # print(self.target[0])
        
        # normalize data
        if mode == 'train':
            self.transform = train_transform
        else:
            self.transform = test_transform

    def __getitem__(self, index):
        x = self.data[index]
        # x = np.pad(x,[(0,0),(2,2),(2,2)],mode='constant',constant_values=0)
        y = self.target[index]  
        x = np.uint(x.reshape(12,48)*255)
        # print(x.shape,x.max(),x.min(),x.std())
        x = Image.fromarray(x, 'L')
        # x_p = np.asarray(x)
        # x.save(fp=str(index)+'.png')
        # print(x_p.shape,x_p.max(),x_p.min())

        # plot_x = np.asarray(self.transform(x))
        # plt.pcolormesh(plot_x.reshape(12,48))
        # plt.show()
        x = self.transform(x)
        # print(x.shape,x.max(),x.min(),x.std())
        return x, y
        # return x, y
    
    def __len__(self):
        return len(self.data)

class SPEC(Dataset):
    def __init__(self,mode,opt,sample_per_class=None):
        self.data = torch.from_numpy(getattr(readSPEC(opt,sample_per_class),'{}_data'.format(mode))).float() # .permute(0,3,1,2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readSPEC(opt,sample_per_class),'{}_labels'.format(mode))).long()
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

def load_spec(opt='_exp1',batch_size=32,test_batch_size = 32, sample_per_class=None):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(SPEC('train',opt,sample_per_class),
    batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(SPEC('test',opt),
    batch_size=test_batch_size, shuffle=False, **kwargs)
    # valid_loader = torch.utils.data.DataLoader(GTSRB('val'),
    # batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader



class SPEC_16x48(Dataset):
    def __init__(self,mode,opt,sample_per_class=None):
        self.data = torch.from_numpy(getattr(readSPEC(opt,sample_per_class),'{}_data'.format(mode))).float() # .permute(0,3,1,2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readSPEC(opt,sample_per_class),'{}_labels'.format(mode))).long()
        # print(self.target[0])
        
        # normalize data
        # self.transform = transforms.Normalize((0.5,), (0.5,))

    def __getitem__(self, index):
        x = self.data[index]
        p3d = (0,0,2,2,0,0)
        x = F.pad(x,p3d,"constant",0)
        y = self.target[index]        
        return x, y
        # return x, y
    
    def __len__(self):
        return len(self.data)

def load_spec_16x48(opt='_exp1',batch_size=32,test_batch_size = 32, sample_per_class=None):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(SPEC_16x48('train',opt,sample_per_class),
    batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(SPEC_16x48('test',opt),
    batch_size=test_batch_size, shuffle=False, **kwargs)
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

def get_class_number(fpost):
    data_root = '/data3/wenjie/echo_localization/DATA_FOLDER/cnn_data/spec'
    fpath = os.path.join(data_root,'test_{}.p'.format(fpost))
    with open(fpath, 'rb') as rfile:
        dataset =  pickle.load(rfile)

    labels = np.array(dataset['labels'])

    unique, _ = np.unique(labels, return_counts=True)
    # print(unique)

    return len(unique)
