from enum import Enum
import hashlib
import math
import os
import random
import re

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import csv
from sklearn.neighbors import KernelDensity

from manage_audio import AudioPreprocessor

class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class ConfigType(Enum):
    CNN_TRAD_POOL2 = "cnn-trad-pool2" # default full model (TF variant)
    CNN_ONE_STRIDE1 = "cnn-one-stride1" # default compact model (TF variant)
    CNN_ONE_FPOOL3 = "cnn-one-fpool3"
    CNN_ONE_FSTRIDE4 = "cnn-one-fstride4"
    CNN_ONE_FSTRIDE8 = "cnn-one-fstride8"
    CNN_TPOOL2 = "cnn-tpool2"
    CNN_TPOOL3 = "cnn-tpool3"
    CNN_TSTRIDE2 = "cnn-tstride2"
    CNN_TSTRIDE4 = "cnn-tstride4"
    CNN_TSTRIDE8 = "cnn-tstride8"
    RES15 = "res15"
    RES26 = "res26"
    RES8 = "res8"
    RES15_NARROW = "res15-narrow"
    RES8_NARROW = "res8-narrow"
    RES26_NARROW = "res26-narrow"

def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("res"):
        return SpeechResModel
    else:
        return SpeechModel

def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    return _configs[conf]

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class SpeechResModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)

class SpeechModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]

        conv1_size = config["conv1_size"] # (time, frequency)
        conv1_pool = config["conv1_pool"]
        conv1_stride = tuple(config["conv1_stride"])
        dropout_prob = config["dropout_prob"]
        width = config["width"]
        height = config["height"]
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        tf_variant = config.get("tf_variant")
        self.tf_variant = tf_variant
        if tf_variant:
            truncated_normal(self.conv1.weight.data)
            self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(conv1_pool)

        x = Variable(torch.zeros(1, 1, height, width), volatile=True)
        x = self.pool1(self.conv1(x))
        conv_net_size = x.view(1, -1).size(1)
        last_size = conv_net_size

        if "conv2_size" in config:
            conv2_size = config["conv2_size"]
            conv2_pool = config["conv2_pool"]
            conv2_stride = tuple(config["conv2_stride"])
            n_featmaps2 = config["n_feature_maps2"]
            self.conv2 = nn.Conv2d(n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride)
            if tf_variant:
                truncated_normal(self.conv2.weight.data)
                self.conv2.bias.data.zero_()
            self.pool2 = nn.MaxPool2d(conv2_pool)
            x = self.pool2(self.conv2(x))
            conv_net_size = x.view(1, -1).size(1)
            last_size = conv_net_size
        if not tf_variant:
            self.lin = nn.Linear(conv_net_size, 32)

        if "dnn1_size" in config:
            dnn1_size = config["dnn1_size"]
            last_size = dnn1_size
            if tf_variant:
                self.dnn1 = nn.Linear(conv_net_size, dnn1_size)
                truncated_normal(self.dnn1.weight.data)
                self.dnn1.bias.data.zero_()
            else:
                self.dnn1 = nn.Linear(32, dnn1_size)
            if "dnn2_size" in config:
                dnn2_size = config["dnn2_size"]
                last_size = dnn2_size
                self.dnn2 = nn.Linear(dnn1_size, dnn2_size)
                if tf_variant:
                    truncated_normal(self.dnn2.weight.data)
                    self.dnn2.bias.data.zero_()
        self.output = nn.Linear(last_size, n_labels)
        if tf_variant:
            truncated_normal(self.output.weight.data)
            self.output.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1))) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        if hasattr(self, "conv2"):
            x = F.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
            x = self.dropout(x)
            x = self.pool2(x)
        x = x.view(x.size(0), -1) # shape: (batch, o3)
        if hasattr(self, "lin"):
            x = self.lin(x)
        if hasattr(self, "dnn1"):
            x = self.dnn1(x)
            if not self.tf_variant:
                x = F.relu(x)
            x = self.dropout(x)
        if hasattr(self, "dnn2"):
            x = self.dnn2(x)
            x = self.dropout(x)
        return self.output(x)

class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2

# wenjie added to compute transfet functions
# implemented by wj, arguments are hardcoded
def fft_to_obtain_tf(x1,x2):
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160
    
    r1 = 4*sample_rate - 1
    r2 = 304*sample_rate
    
    y1 = librosa.core.load(x1, sr=16000)[0]
    y2 = librosa.core.load(x2, sr=16000)[0]

    # STFT
    FFT1 = librosa.stft(y1[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT1_abs = np.abs(FFT1)
    FFT1_mean = np.mean(FFT1_abs,axis=1)
    
    FFT2 = librosa.stft(y2[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT2_abs = np.abs(FFT2)
    FFT2_mean = np.mean(FFT2_abs,axis=1)
    
    TF = np.divide(FFT2_mean,FFT1_mean)
    return TF[:,None]/TF.max()

def fft_to_obtain_tf_with_time(x1,x2,s_t,e_t):
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160
    
    r1 = s_t*sample_rate
    r2 = e_t*sample_rate - 1
    
    y1 = librosa.core.load(x1, sr=16000)[0]
    y2 = librosa.core.load(x2, sr=16000)[0]
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()

    # STFT
    FFT1 = librosa.stft(y1[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    # print("FFT1 shape: ",FFT1.shape)
    FFT1_abs = np.abs(FFT1)
    FFT1_mean = np.mean(FFT1_abs,axis=1)
    
    FFT2 = librosa.stft(y2[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT2_abs = np.abs(FFT2)
    FFT2_mean = np.mean(FFT2_abs,axis=1)
    
    TF = np.divide(FFT2_mean,FFT1_mean)
    return TF[:,None]/TF.max()

def obtain_tf_USB_with_time(x1,x2,s_t,e_t):
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160

    r1 = s_t*sample_rate
    r2 = e_t*sample_rate - 1
    
    y1 = librosa.core.load(x1, sr=16000)[0]
    y2 = librosa.core.load(x2, sr=16000)[0]
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()    
    # STFT
    FFT1 = librosa.stft(y1[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT1_abs = np.abs(FFT1)
    FFT1_mean = np.mean(FFT1_abs,axis=1)
    
    FFT2 = librosa.stft(y2[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT2_abs = np.abs(FFT2)
    FFT2_mean = np.mean(FFT2_abs,axis=1)
    
    TF = np.divide(FFT2_mean,FFT1_mean)
    TF[:5] = 0
    return TF[:,None]/TF.max()

# no normalization
def obtain_tf_no_norm(x1,x2):
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160

    r1 = 4*sample_rate - 1
    r2 = 304*sample_rate
    
    y1 = librosa.core.load(x1, sr=16000)[0]
    y2 = librosa.core.load(x2, sr=16000)[0]
    
    # STFT
    FFT1 = librosa.stft(y1[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT1_abs = np.abs(FFT1)
    FFT1_mean = np.mean(FFT1_abs,axis=1)
    
    FFT2 = librosa.stft(y2[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT2_abs = np.abs(FFT2)
    FFT2_mean = np.mean(FFT2_abs,axis=1)
    
    TF = np.divide(FFT2_mean,FFT1_mean)
    return TF

def obtain_tf_usb(x1,x2):
    tf = obtain_tf_no_norm(x1,x2)
    tf[:5] = 0
    return tf[:,None]/tf.max()

def obtain_tf_no_norm_with_time(x1,x2,s):
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160

    r1 = 0
    r2 = int(s*sample_rate) 
    
    y1 = librosa.core.load(x1, sr=16000)[0]
    y2 = librosa.core.load(x2, sr=16000)[0]
    
    # STFT
    FFT1 = librosa.stft(y1[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT1_abs = np.abs(FFT1)
    FFT1_mean = np.mean(FFT1_abs,axis=1)
    
    FFT2 = librosa.stft(y2[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT2_abs = np.abs(FFT2)
    FFT2_mean = np.mean(FFT2_abs,axis=1)
    
    TF = np.divide(FFT2_mean,FFT1_mean)
    return TF

def obtain_tf_usb_with_time(x1,x2,s):
    tf = obtain_tf_no_norm_with_time(x1,x2,s)
    tf[:5] = 0
    return tf[:,None]/tf.max()

def stft_ave_no_DC(x):
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160
    
    normalize = 'True'
    r1 = 4*sample_rate - 1
    r2 = 304*sample_rate
    
    y = librosa.core.load(x, sr=16000)[0]
    y -= y.mean()
    
    # STFT
    FFT = librosa.stft(y[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT_abs = np.abs(FFT)
    FFT_mean = np.mean(FFT_abs,axis=1)

    return FFT_mean[:,None]/FFT_mean.max()

###
def fft_to_obtain_tf_time(x1,x2,s):
    # s in seconds
    sample_rate = 16000
    window = 'hann'
    n_fft = 480
    win_length = n_fft
    hop_length = 160
    
    r1 = 0
    r2 = int(s*sample_rate)
    print("r2 is ",r2)
    
    y1 = librosa.core.load(x1, sr=16000)[0]
    y2 = librosa.core.load(x2, sr=16000)[0]
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()

    # STFT
    FFT1 = librosa.stft(y1[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    print("FFT1 shape: ",FFT1.shape)
    FFT1_abs = np.abs(FFT1)
    FFT1_mean = np.mean(FFT1_abs,axis=1)
    
    FFT2 = librosa.stft(y2[r1:r2], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    FFT2_abs = np.abs(FFT2)
    FFT2_mean = np.mean(FFT2_abs,axis=1)
    
    TF = np.divide(FFT2_mean,FFT1_mean)
    return TF[:,None]/TF.max()


def get_tf_phyaug(t):
    flist = [
        "../data/recorded_dataset/white-noise/white-noise-true-20min.wav",
        "../data/recorded_dataset/white-noise/ATR-white.wav",
        "../data/recorded_dataset/white-noise/clipon-white.wav",
        "../data/recorded_dataset/white-noise/maono-white.wav",
        "../data/recorded_dataset/white-noise/USB-white.wav",
        "../data/recorded_dataset/white-noise/USBplug-white.wav",
        "../data/recorded_dataset/white-noise/ATR-silence.wav",
        "../data/recorded_dataset/white-noise/clipon-silence.wav",
        "../data/recorded_dataset/white-noise/maono-silence.wav",
        "../data/recorded_dataset/white-noise/USB-silence.wav",
        "../data/recorded_dataset/white-noise/USBplug-silence.wav"
        ]
    tf_atr = fft_to_obtain_tf_time(flist[0],flist[1],t) + stft_ave_no_DC(flist[6])
    tf_clipon = fft_to_obtain_tf_time(flist[0],flist[2],t) + stft_ave_no_DC(flist[7])
    tf_maono = fft_to_obtain_tf_time(flist[0],flist[3],t) + stft_ave_no_DC(flist[8])
    tf_USB = obtain_tf_usb_with_time(flist[0],flist[4],t) + stft_ave_no_DC(flist[9])
    tf_USBplug = obtain_tf_usb_with_time(flist[0],flist[5],t) + stft_ave_no_DC(flist[10])
    return [tf_atr,tf_clipon,tf_maono,tf_USB,tf_USBplug]


def get_kde_meetroom_loc2_45cm():
    flist = [
        "/data2/wenjie/deepspeech.pytorch/data/audio/TR2_loc1/white-noise-true-20min.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/ATR-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/clipon-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/maono-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/USB-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/USBplug-white.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/ATR-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/clipon-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/maono-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/USB-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/USBplug-silence.wav"
        ]
    tf_atr = fft_to_obtain_tf_with_time(flist[0],flist[1],0,5).reshape(-1,1)
    tf_maono = fft_to_obtain_tf_with_time(flist[0],flist[2],0,5).reshape(-1,1)
    tf_clipon = (obtain_tf_USB_with_time(flist[0],flist[3],0,5) + stft_ave_no_DC(flist[7])).reshape(-1,1) # obtain transfer function
    tf_USB = (obtain_tf_USB_with_time(flist[0],flist[4],0,5) + stft_ave_no_DC(flist[9])).reshape(-1,1) # obtain transfer function
    tf_USBplug = (obtain_tf_USB_with_time(flist[0],flist[5],0,5) + stft_ave_no_DC(flist[10])).reshape(-1,1) # obtain transfer function
    for t in range(5,300,5):
        atr_temp = fft_to_obtain_tf_with_time(flist[0],flist[1],t,t+5)
        tf_atr = np.concatenate((tf_atr,
                                atr_temp.reshape(-1,1)),
                                axis=1)
        maono_temp = fft_to_obtain_tf_with_time(flist[0],flist[2],t,t+5)
        tf_maono = np.concatenate((tf_maono,
                                maono_temp.reshape(-1,1)),
                                axis=1)
        clipon_temp = obtain_tf_USB_with_time(flist[0],flist[3],t,t+5) + stft_ave_no_DC(flist[7])
        tf_clipon = np.concatenate((tf_clipon,
                                clipon_temp.reshape(-1,1)),
                                axis=1)
        USB_temp = obtain_tf_USB_with_time(flist[0],flist[4],t,t+5) + stft_ave_no_DC(flist[9])
        tf_USB = np.concatenate((tf_USB,
                                USB_temp.reshape(-1,1)),
                                axis=1)
        USBplug_temp = obtain_tf_USB_with_time(flist[0],flist[5],t,t+5) + stft_ave_no_DC(flist[10])
        tf_USBplug = np.concatenate((tf_USBplug,
                                USBplug_temp.reshape(-1,1)),
                                axis=1)
        print("t = ",t)
    kde_atr = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_atr.T)
    kde_maono = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_maono.T)
    kde_clipon = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_clipon.T)
    kde_USB = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_USB.T)
    kde_USBplug = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_USBplug.T)
    return [kde_atr,kde_maono,kde_clipon,kde_USB,kde_USBplug]

def get_tf_meetroom_loc2_45cm_with_time_no_DC(t):
    flist = [
        "/data2/wenjie/deepspeech.pytorch/data/audio/TR2_loc1/white-noise-true-20min.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/ATR-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/clipon-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/maono-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/USB-white.wav",
        "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd/white-noise/convert/USBplug-white.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/ATR-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/clipon-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/maono-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/USB-silence.wav",
        "/data2/wenjie/deepspeech.pytorch/librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/USBplug-silence.wav"
        ]
    tf_atr = fft_to_obtain_tf_time(flist[0],flist[1],t) # + stft_ave_no_DC(flist[6])
    tf_clipon = fft_to_obtain_tf_time(flist[0],flist[2],t)#  + stft_ave_no_DC(flist[7])
    tf_maono = fft_to_obtain_tf_time(flist[0],flist[3],t) #+ stft_ave_no_DC(flist[8])
    tf_USB = obtain_tf_usb_with_time(flist[0],flist[4],t) #+ stft_ave_no_DC(flist[9])
    tf_USBplug = obtain_tf_usb_with_time(flist[0],flist[5],t)# + stft_ave_no_DC(flist[10])
    return [tf_atr,tf_clipon,tf_maono,tf_USB,tf_USBplug]

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.audio_processor = AudioPreprocessor(n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"], hop_ms=10, tf = [])
        self.audio_preprocess_type = config["audio_preprocess_type"]

    @staticmethod
    def default_config():
        config = {}
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.1
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        # config["wanted_words"] = ["yes", "no","left","right"]
        # config["wanted_words"] = ["yes", "no","left","right"]
        config["wanted_words"] = ["yes", "no","left","right","up","down","on","off","stop","go"]
        # config["wanted_words"] = ["house"]

        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_original/mic2mic"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_usbplug"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_atr/audio_house"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_maono/house_audio"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_clipon/house_audio"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_USB/house_audio"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/mic2mic_USBplug/house_audio"

        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/original"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/original_testset"
        # TR
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/TR1/ATR"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/TR1/maono"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/TR1/clipon"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/TR1/USB"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/TR1/USBplug"
        # meeting room loc2-30cm
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-30cm/ATR"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-30cm/maono"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-30cm/clipon"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-30cm/USB"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-30cm/USBplug"
        # meeting room loc2-45cm
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm/ATR"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm/maono"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm/clipon"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm/USB"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm/USBplug"
        # meeting room loc2-45cm-10cmd
        config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd-v1/ATR"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd-v1/maono"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd-v1/clipon"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd-v1/USB"
        # config["data_folder"] = "/data1/wenjie/github/honk/speech_dataset_wenjie/meetingroom/loc2-45cm-10cmd-v1/USBplug"
        config["audio_preprocess_type"] = "MFCCs"
        return config

    def collate_fn(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data).reshape(1, -1, 40))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def collate_spectrogram(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_spectrogram(audio_data).reshape(1,241,101)) # .reshape(1, -1, 40))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)
    
    def collate_fn_with_tf(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs_with_tf(audio_data).reshape(1, -1, 40))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def collate_fn_with_kde(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs_with_kde(audio_data).reshape(1, -1, 40))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def collate_fn_inverse_tf(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs_inverse_tf(audio_data).reshape(1, -1, 40))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def load_audio(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() < 0.7 or not self.set_type == DatasetType.TRAIN:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
            self._file_cache[example] = data
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        if self.set_type == DatasetType.TRAIN:
            data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        self._audio_cache[example] = data
        return data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            # check folder to see if folder name is in wanted words, if not treat as unknown
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                # append background noise
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                # append unkown lists
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                # print("filename is: ",filename, "hashname: ",hashname, "bucket: ",bucket)
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                    # with open('test_set_4cmd_original_testset.csv', 'a', newline = '') as f:
                    #     wr = csv.writer(f, dialect='excel')
                    #     wr.writerow([path_name,filename])
                    # f.close()
                else:
                    tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None, silence=True), 0
        return self.load_audio(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence

_configs = {
    ConfigType.CNN_TRAD_POOL2.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=64,
        n_feature_maps2=64, conv1_size=(20, 8), conv2_size=(10, 4), conv1_pool=(2, 2), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True),
    ConfigType.CNN_ONE_STRIDE1.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=186,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128, tf_variant=True),
    ConfigType.CNN_TSTRIDE2.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=78,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(9, 4), conv1_pool=(1, 3), conv1_stride=(2, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TSTRIDE4.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=100,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(5, 4), conv1_pool=(1, 3), conv1_stride=(4, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TSTRIDE8.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=126,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(5, 4), conv1_pool=(1, 3), conv1_stride=(8, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TPOOL2.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=94,
        n_feature_maps2=94, conv1_size=(21, 8), conv2_size=(6, 4), conv1_pool=(2, 3), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TPOOL3.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=94,
        n_feature_maps2=94, conv1_size=(15, 8), conv2_size=(6, 4), conv1_pool=(3, 3), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FPOOL3.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=54,
        conv1_size=(101, 8), conv1_pool=(1, 3), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FSTRIDE4.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=186,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 4), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FSTRIDE8.value: dict(dropout_prob=0.5, height=101, width=40, n_labels=12, n_feature_maps1=336,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 8), dnn1_size=128, dnn2_size=128),
    ConfigType.RES15.value: dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=45),
    ConfigType.RES8.value: dict(n_labels=12, n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26.value: dict(n_labels=12, n_layers=24, n_feature_maps=45, res_pool=(2, 2), use_dilation=False),
    ConfigType.RES15_NARROW.value: dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19),
    ConfigType.RES8_NARROW.value: dict(n_labels=12, n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26_NARROW.value: dict(n_labels=12, n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False)
}
