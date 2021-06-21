# from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os
import utils
from utils import get_pretrained_model,train,imshow_tensor,save_checkpoint,load_checkpoint,process_image,predict,random_test_image,accuracy,Evaluate
import sys
# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer
# from wand.image import Image
import wand

# Visualizations
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['font.size'] = 14

# Printing out all outputs
# InteractiveShell.ast_node_interactivity = 'all'

# Location of data
datadir = '/data1/wenjie/github/pytorch_challenge/data/caltech101_fisheye_fisheye_0p2_0p2_0p01/'
traindir = datadir + 'train/'
validdir = datadir + 'valid/'
testdir = datadir + 'test/'

# Change to fit hardware
batch_size = 64

# Whether to train on a gpu
train_on_gpu = cuda.is_available()

print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

device_name = cuda.get_device_name()
print(device_name)

# DATA_ROOT = '/data1/wenjie/github/pytorch_challenge/data/'
# CALTECH101_ORIGINAL = os.path.join(DATA_ROOT, '101_ObjectCategories')
# CALTECH101_ROOT = os.path.join(DATA_ROOT, 'caltech101_fisheye_Augmentation')
#split data into train, val and test set
# from utils import split_dataset_folder
# split_dataset_folder(CALTECH101_ORIGINAL, CALTECH101_ROOT)

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets from each folder
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)

print("class numbers: ",len(data['train'].classes))

n_classes = len(data['train'].classes)

# This should return the same as the pretrained model with the custom classifier. In the case of resnet, we replace the `fc` layers with the same classifier.
# 
# The `torchsummary` library has a helpful function called `summary` which summarizes our model.

model = get_pretrained_model('resnet50',train_on_gpu,n_classes,multi_gpu)
# if multi_gpu:
#     summary(
#         model.module,
#         input_size=(3, 224, 224),
#         batch_size=batch_size,
#         device='cuda')
# else:
#     summary(
#         model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')


model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())[:10]

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

save_file_name = 'resnet50-caltech_CDA.pth'
checkpoint_path = 'resnet50-caltech_CDA.pth'

model, history = train(
    train_on_gpu,
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=10,
    n_epochs=40,
    print_every=2)

save_checkpoint(model, checkpoint_path, multi_gpu)


model, optimizer = load_checkpoint(checkpoint_path, multi_gpu, train_on_gpu)

# Evaluate the model on all the training data
original_accuracy,_ = Evaluate(model, dataloaders['test'], criterion, train_on_gpu)
# results = evaluate(model, dataloaders['test'], criterion, n_classes, train_on_gpu)
print("test accuracy on original set is: ", original_accuracy)

## evaluate on fisheye dataset
datadir = '/data1/wenjie/github/pytorch_challenge/data/caltech101_fisheye_fisheye_0p2_0p2_0p01/'
traindir = datadir + 'train/'
validdir = datadir + 'valid/'
testdir = datadir + 'test/'

# Datasets from each folder
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=1, shuffle=True)
}

fisheye_accuracy,_ = Evaluate(model, dataloaders['test'], criterion, train_on_gpu)
# results = evaluate(model, dataloaders['test'], criterion, n_classes, train_on_gpu)
print("test accuracy on fisheye set is: ", fisheye_accuracy)
