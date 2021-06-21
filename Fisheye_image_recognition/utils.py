# utils file

import subprocess
import os
import sys
import glob
import json
import shutil
from PIL import Image
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torchvision
from torchvision import datasets, transforms
from torchvision import models
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torch.autograd import Variable
from torchvision import models
from torch.utils.data import DataLoader
import torchnet as tnt
import time
from scipy.interpolate import interp1d
import subprocess
import requests
from urllib.request import urlretrieve
import tarfile
import copy
from timeit import default_timer as timer
import pandas as pd


def get_number_processors():
    """Get the number of processors in a CPU.
    Returns:
        num (int): Number of processors.
    Examples:
        >>> get_number_processors()
        4
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing #force exception in case mutiprocessing is not installed
        num = multiprocessing.cpu_count()
    return num


def get_gpu_name():
    """Get the GPUs in the system
    Examples:
        >>> get_gpu_name()
        ['Tesla M60', 'Tesla M60', 'Tesla M60', 'Tesla M60']
        
    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_gpu_memory():
    """Get the memory of the GPUs in the system
    Examples:
        >>> get_gpu_memory()
        ['8123 MiB', '8123 MiB', '8123 MiB', '8123 MiB']

    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").replace('\r','').split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)

        
def get_cuda_version():
    """Get the CUDA version
    Examples:
        >>> get_cuda_version()
        'CUDA Version 8.0.61'

    """
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
    elif sys.platform == 'linux':
        path = '/usr/local/cuda/version.txt'
        if os.path.isfile(path):
            with open(path, 'r') as f:
                data = f.read().replace('\n','')
            return data
        else:
            return "No CUDA in this machine"
    elif sys.platform == 'darwin':
        raise NotImplementedError("Find a Mac with GPU and implement this!")
    else:
        raise ValueError("Not in Windows, Linux or Mac")
        
    
def format_dictionary(dct, indent=4):
    """Formats a dictionary to be printed
    Parameters:
        dct (dict): Dictionary.
        indent (int): Indentation value.
    Returns:
        result (str): Formatted dictionary ready to be printed
    Examples:
        >>> dct = {'bkey':1, 'akey':2}
        >>> print(format_dictionary(dct))
        {
            "akey": 2,
            "bkey": 1
        }
    """
    return json.dumps(dct, indent=indent, sort_keys=True)



def get_filenames_in_folder(folderpath):
    """ Return the files names in a folder.
    Parameters:
        folderpath (str): folder path
    Returns:
        number (list): list of files
    Examples:
        >>> get_filenames_in_folder('C:/run3x/codebase/python/minsc')
        ['paths.py', 'system_info.py', '__init__.py']

    """
    names = [os.path.basename(x) for x in glob.glob(os.path.join(folderpath, '*'))]
    return sorted(names)


def get_files_in_folder_recursively(folderpath):
    """ Return the files inside a folder recursivaly.
    Parameters:
        folderpath (str): folder path
    Returns:
        filelist (list): list of files
    Examples:
        >>> get_files_in_folder_recursively(r'C:\\run3x\\codebase\\command_line')
        ['linux\\compress.txt', 'linux\\paths.txt', 'windows\\resources_management.txt']
    """
    if folderpath[-1] != os.path.sep: #Add final '/' if it doesn't exist
        folderpath += os.path.sep
    names = [x.replace(folderpath,'') for x in glob.iglob(folderpath+'/**', recursive=True) if os.path.isfile(x)]
    return sorted(names)



def _make_directory(directory):
    """Make a directory"""
    if not os.path.isdir(directory):
        os.makedirs(directory)

        
def _create_sets_folders(root_folder, sets_names, target_folder):
    """Create folder structure"""
    for s in sets_names:
        dest = os.path.join(root_folder, s, target_folder)
        _make_directory(dest)
          
                
def split_list(py_list, perc_size, shuffle=False):
    """Split a list in weighted chunks
    Parameters:
        py_list (list): A list of elements.
        perc_size (list): The percentual size of each chunk size.
        shuffle (bool): Shuffle the list or not
    Returns:
        result_list (list of list): A list of lists with the chunks.
    Examples:
        >>> split_list(list(range(7)),[0.47,0.33,0.2])
        [[0, 1, 2], [3, 4, 5], [6]]
        >>> split_list(list(range(10)),[0.6,0.4], True)
        [[1, 2, 3, 6, 9, 5], [4, 8, 0, 7]]

    """
    assert sum(perc_size) == 1, "Percentage sizes do not sum to 1"
    l = py_list[:]
    if shuffle:
        random.shuffle(l)
    # Turn percentages into values between 0 and 1
    splits = np.cumsum(perc_size)

    # Split doesn't need last percent, it will just take what is left
    splits = splits[:-1]

    # Turn values into indices
    splits *= len(l)

    # Turn double indices into integers.
    splits = splits.round().astype(np.int)

    return [list(chunks) for chunks in np.split(l, splits)]


def split_dataset_folder(root_folder, dest_folder, sets_names=['train','valid','test'], sets_sizes=[0.5,0.25,0.25], shuffle=False, verbose=True):
    """Split the folders in a dataset to pytorch format. If the intial format is:
    --class1
    ----img1.jpg
    ----img2.jpg
    --class2
    ----img1.jpg
    ----img2.jpg
    It transforms it into:
    --train
    ----class1
    ------img1.jpg
    ----class2
    ------img1.jpg
    --val
    ----class1
    ------img2.jpg
    ----class2
    ------img2.jpg   

    """
    assert sum(sets_sizes) == 1, "Data set sizes do not sum to 1"
    for folder in get_filenames_in_folder(root_folder):
        if verbose: print("Folder: ", folder)
        _create_sets_folders(dest_folder, sets_names, folder)
        files = get_filenames_in_folder(os.path.join(root_folder, folder))
        files_split = split_list(files, sets_sizes, shuffle)
        for split, set_name in zip(files_split, sets_names):
            for f in split:
                orig = os.path.join(root_folder, folder, f)
                dest = os.path.join(dest_folder, set_name, folder)
                if verbose: print("Copying {} into {}".format(orig, dest))
                shutil.copy2(orig, dest)

                
def convert_image_dataset_to_grayscale(root_folder, dest_folder, verbose=True):
    """Convert all the images from a dataset in disk to grayscale"""
    files = get_files_in_folder_recursively(root_folder)
    for f in files:
        filename = os.path.join(root_folder, f)
        if verbose: print("Converting {} to grayscale".format(filename))
        img = Image.open(filename)
        img_gray = img.convert('L')
        dest = os.path.join(dest_folder, f)
        try:
            img_gray.save(dest)
        except FileNotFoundError as e:
            if verbose: print(e)
            path = os.path.dirname(dest)
            if verbose: print("Creating folder {}".format(path))
            os.makedirs(path)
            img_gray.save(dest)
            
            
def create_dataset(data_dir, batch_size=32, sets=['train', 'val'], verbose=True):
    """Create a dataset object given the path. On data_dir there should be a train and validation folder
    and in each of them there should be the folders containing the data. One folder for each class
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=get_number_processors()) 
                   for x in sets}

    if verbose:
        dataset_sizes = {x: len(image_datasets[x]) for x in sets}
        class_names = dataloaders[sets[0]].dataset.class_to_idx
        print("There are {} clases in the dataset: {}".format(len(class_names), format_dictionary(class_names)))
        print("Sets sizes: ", format_dictionary(dataset_sizes))
        for x in sets:   
            c = Counter(item[1] for item in image_datasets[x])
            c = dict(c)
            print("Number of items in set {}: {}".format(x, c))
    return dataloaders


def plot_pytorch_data_stream(dataobject, max_images=8, title=True):
    """Plot a batch of images"""
    inputs, classes = next(iter(dataobject))  
    if max_images > dataobject.batch_size:
        max_images = dataobject.batch_size
        print("Plotting only {} images, which is the batch size".format(max_images))
    inputs = inputs[:max_images,:,:,:]
    classes = classes[:max_images]
    out = torchvision.utils.make_grid(inputs)
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title:
        class_names = dataobject.dataset.classes
        names = [class_names[x] for x in classes]
        plt.title(names)

        
def finetune(dataloaders, model_name, sets, num_epochs, num_gpus, lr, momentum, lr_step, lr_epochs, verbose=False):
    """Finetune all the layers of a model using a dataset loader. """
    #Class adaptation
    num_class = len(dataloaders[sets[0]].dataset.class_to_idx)
    model_ft = models.__dict__[model_name](pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    
    #gpus
    if num_gpus > 1: 
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()
    
    #loss
    criterion = nn.CrossEntropyLoss()

    # All parameters are being optimized
    optimizer = SGD(model_ft.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of lr_step every lr_epochs epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_epochs, gamma=lr_step)
    model_ft = train_model(dataloaders, model_ft, sets, criterion, optimizer, exp_lr_scheduler, 
                           num_epochs=num_epochs, verbose=verbose)
    return model_ft


def freeze_and_train(dataloaders, model_name, sets, num_epochs, num_gpus, lr, momentum, lr_step, lr_epochs, verbose=False):
    """Freezes all layers but the last one and train the last layer using a dataset loader"""
    #Class adaptation
    num_class = len(dataloaders[sets[0]].dataset.class_to_idx)
    model_conv = models.__dict__[model_name](pretrained=True)
    for param in model_conv.parameters(): #params have requires_grad=True by default
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_class)
    
    #gpus
    if num_gpus > 1: 
        model_conv = nn.DataParallel(model_conv)
    model_conv = model_conv.cuda()
    
    #loss
    criterion = nn.CrossEntropyLoss()

    # Only parameters of final layer are being optimized 
    if num_gpus > 1:
        params = model_conv.module.fc.parameters()
    else:
        params = model_conv.fc.parameters()
    optimizer = SGD(params, lr=lr, momentum=momentum)

    # Decay LR by a factor of lr_step every lr_epochs epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_epochs, gamma=lr_step)
    model_conv = train_model(dataloaders, model_conv, sets, criterion, optimizer, exp_lr_scheduler, 
                             num_epochs=num_epochs, verbose=verbose)
    return model_conv


def train_model(dataloaders, model, sets, criterion, optimizer, scheduler, num_epochs=25, verbose=False):
    """Train a pytorch model"""
    since = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in sets}
    best_model_wts = model.state_dict()
    best_acc = 0.0
    num_classes = len(dataloaders[sets[0]].dataset.classes)
    confusion_matrix = tnt.meter.ConfusionMeter(num_classes, normalized=False)
    metrics = {'train_acc':[],'val_acc':[],'train_loss':[],'val_loss':[], 'cm':[]}
    for epoch in range(num_epochs):
        if verbose:
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in sets:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                confusion_matrix.reset()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                else:
                    confusion_matrix.add(outputs.data, labels.data)
                    
                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)
            #metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / float(dataset_sizes[phase])
            if verbose: print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                metrics['train_acc'].append(epoch_acc)
                metrics['train_loss'].append(epoch_loss)
            else:
                metrics['val_acc'].append(epoch_acc)
                metrics['val_loss'].append(epoch_loss)
                cm = confusion_matrix.value().copy()
                metrics['cm'].append(cm)
                
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def available_models():
    """Return available pytorch models, callable using `models.__dict__[name]`"""
    model_names = sorted(name for name in models.__dict__  if name.islower() and not name.startswith("__") and 
                         callable(models.__dict__[name]))
    return model_names


def plot_metrics(metrics, title=None):
    """Plot metrics from training. metrics is a dict containing 'train_acc', 'val_acc', 'train_loss' and
    'val_loss', each of them contains the metrics values in a list"""
    max_epochs = len(metrics['train_acc']) + 1
    epochs = range(1, max_epochs)
    epochs_dx = np.linspace(epochs[0], epochs[-1], num=max_epochs*4, endpoint=True)
    s_train_acc = interp1d(epochs, metrics['train_acc'], kind='cubic')    
    s_val_acc = interp1d(epochs, metrics['val_acc'], kind='cubic')    
    s_train_loss = interp1d(epochs, metrics['train_loss'], kind='cubic')    
    s_val_loss = interp1d(epochs, metrics['val_loss'], kind='cubic')    

    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(right=2, top=0.85)
    if title is not None:
        st = fig.suptitle(title, fontsize=16)
        st.set_x(1)
    ax[0].plot(epochs, metrics['train_acc'], 'b.', label='train')
    ax[0].plot(epochs_dx, s_train_acc(epochs_dx), 'b')
    ax[0].plot(epochs, metrics['val_acc'], 'g.', label='val')
    ax[0].plot(epochs_dx, s_val_acc(epochs_dx), 'g')
    ax[0].legend( loc="lower right")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].xaxis.set_major_locator(MultipleLocator(1))#only integers in axis multiples of 1
    
    ax[1].plot(epochs, metrics['train_loss'], 'b.', label='train')
    ax[1].plot(epochs_dx, s_train_loss(epochs_dx), 'b')
    ax[1].plot(epochs, metrics['val_loss'], 'g.', label='val')
    ax[1].plot(epochs_dx, s_val_loss(epochs_dx), 'g')
    ax[1].legend(loc="upper right")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    plt.show()
    
    
def _list_lsun_categories(tag):
    """List LSUN categories"""
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = requests.get(url)
    return json.loads(f.text)


def _download_lsun(out_dir, category, set_name, tag):
    """Download a specific category of LSUN"""
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    _make_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)  
    
    
def download_lsun_dataset(out_dir):
    """Download LSUN dataset and create pytorch folder structure
    source: https://github.com/fyu/lsun
    """
    tag = 'latest'
    categories = _list_lsun_categories(tag)
    print('Downloading', len(categories), 'categories')
    for category in categories:
        _download_lsun(out_dir, category, 'train', tag)
        _download_lsun(out_dir, category, 'val', tag)
    #_download_lsun(args.out_dir, '', 'test', args.tag)

    
def download_caltech256(out_dir):
    """Download Caltech256 dataset"""
    url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
    if len(os.listdir(out_dir)) != 0:
        print("Dataset already donwloaded in {}".format(out_dir)) 
    else:
        print("Downloading {}".format(url))
        filepath = os.path.join(out_dir, 'delete.me')
        fname, h = urlretrieve(url, filepath)
        print("Extracting files from {}".format(fname))
        with tarfile.open(fname) as tar:
            tar.extractall(path=out_dir)
        os.remove(fname)

def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

# ## Function to Load in Pretrained Model
# 
# We can refactor all that code into a single function that returns a pretrained model. This only accepts the vgg16 or resnet50 at the moment but can be extended to use other models. 


def get_pretrained_model(model_name,
                         train_on_gpu,
                         n_classes,
                         multi_gpu):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1))

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1))

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1))

    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    return model

# # Training
# 
# For training, we iterate through the train `DataLoader`, each time passing one batch through the model. One complete pass through the training data is known as an `epoch`, and we train for a set number of epochs or until early stopping kicks in (more below). After each batch, we calculate the loss (with `criterion(output, targets)`) and then calculate the gradients of the loss with respect to the model parameters with `loss.backward()`. This uses autodifferentiation and backpropagation to calculate the gradients. 
# 
# After calculating the gradients, we call `optimizer.step()` to update the model parameters with the gradients. This is done on every training batch so we are implementing stochastic gradient descent (or rather a version of it with momentum known as Adam). For each batch, we also compute the accuracy for monitoring and after the training loop has completed, we start the validation loop. This will be used to carry out early stopping.
# 
# 
# ## Early Stopping
# 
# Early stopping halts the training when the validation loss has not decreased for a number of epochs. Each time the validation loss does decrease, the model weights are saved so we can later load in the best model. Early stopping is an effective method to prevent overfitting on the training data. If we continue training, the training loss will continue to decrease, but the validation loss will increase because the model is starting to memorize the training data. Early stopping prevents this from happening, and, if we save the model each epoch when the validation loss decreases, we are able to retrieve the model that does best on the validation data.
# 
# Early stopping is implemented by iterating through the validation data at the end of each training epoch and calculating the loss. We use the complete validation data every time and record whether or not the loss has decreased. If it has not for a number of epochs, we stop training, retrieve the best weights, and return them. When in the validation loop, we make sure not to update the model parameters. 
# 
# ### Training Function
# 
# The below function trains the network while monitoring a number of different parameters. We train with early stopping on the validation set. There are a number of parameters that I've tried to explain in the doc string. Hopefully, the comments and background make things somewhat understandable! 

def train(
          train_on_gpu,
          # multi_gpu,
          model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history 

def save_checkpoint(model, path, multi_gpu):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet50':
        if multi_gpu:
            print('saving resnet50 here')
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)
    
def load_checkpoint(path, multi_gpu, train_on_gpu):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer    
    
    
def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor

# ## Function to make predictions
# 
# The next function makes predictions on a single image. It will return the top probabilities and classes. 

def predict(image_path, train_on_gpu, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class

def random_test_image(cat_df, root, testdir):
    """Pick a random test image from the test directory"""
    c = np.random.choice(cat_df['category'])
    root = testdir + c + '/'
    img_path = root + np.random.choice(os.listdir(root))
    return img_path

def display_prediction(image_path, model, topk):
    """Display image and preditions from model"""

    # Get predictions
    img, ps, classes, y_obs = predict(image_path, model, topk)
    # Convert results to dataframe for plotting
    result = pd.DataFrame({'p': ps}, index=classes)

    # Show the image
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 2, 1)
    ax, img = imshow_tensor(img, ax=ax)

    # Set title to be the actual class
    ax.set_title(y_obs, size=20)

    ax = plt.subplot(1, 2, 2)
    # Plot a bar plot of predictions
    result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
    plt.xlabel('Predicted Probability')
    plt.tight_layout()

# The function below computes accuracy for a prediction and a target in terms of topk.
def accuracy(output, target, train_on_gpu, topk=(1, )):
    """Compute the topk accuracy(s)"""
    if train_on_gpu:
        output = output.to('cuda')
        target = target.to('cuda')

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def evaluate(model, test_loader, criterion, train_on_gpu, n_classes, topk=(1, 5)):
    """Measure the performance of a trained PyTorch model

    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:

            # Tensors to gpu
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')

            # Raw model output
            out = model(data)
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(
                    pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(model.idx_to_class[true.item()])
                # Calculate the loss
                print("pred shape: ",pred.shape)
                print("true shape: ",true.shape)
                loss = criterion(pred.view(1, n_classes), true.view(1))
                losses.append(loss.item())
                i += 1

    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()

    return results.reset_index().rename(columns={'index': 'class'})

def Evaluate(model, test_loader, criterion, train_on_gpu, topk=(1, 5)):
    """Measure the performance of a trained PyTorch model

    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    test_acc = 0.0
    test_loss = 0.0

    model.eval()

    total_time = np.array([])
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:

            # Tensors to gpu
            data = torch.randn((len(data),3,1024,1024),dtype=torch.float32)
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')
            # print('data shape:',data.shape)
            # data = torch.randn((len(data),3,224,224),dtype=torch.float32)
            # data.cuda()
            # Raw model output
            s = time.time()
            out = model(data)
            e = time.time()
            eval_time = e - s
            total_time = np.append(total_time,eval_time)
            # print("run time: ",eval_time)

            loss = criterion(out, targets)
            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(out, dim=1)
            correct_tensor = pred.eq(targets.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            test_acc += accuracy.item() * data.size(0)

        test_acc = test_acc / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
    return test_acc,test_loss,total_time


# Given the subtle distinction between `crocodile` and `crocodile_head`, I'd say that this model does pretty well! The ultimate bar in image recognition is human level, and our model is nearly there. 

def display_category(model, category, results, n=4):
    """Display predictions for a category    
    """
    category_results = results.loc[results['class'] == category]
    print(category_results.iloc[:, :6], '/n')

    images = np.random.choice(
        os.listdir(testdir + category + '/'), size=4, replace=False)

    for img in images:
        display_prediction(testdir + category + '/' + img, model, 5)

