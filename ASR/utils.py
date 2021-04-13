import torch
import torch.distributed as dist

from model import DeepSpeech
import librosa
import numpy as np
from sklearn.neighbors import KernelDensity


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device, model_path, use_half):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model

## added by wenjie
# wenjie added to compute transfet functions
# implemented by wj, arguments are hardcoded
def fft_to_obtain_tf(x1,x2):
    sample_rate = 16000
    window = 'hamming'
    n_fft = 320
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
    window = 'hamming'
    n_fft = 320
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
    window = 'hamming'
    n_fft = 320
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
    window = 'hamming'
    n_fft = 320
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
    window = 'hamming'
    n_fft = 320
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
    window = 'hamming'
    n_fft = 320
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
    window = 'hamming'
    n_fft = 320
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

def get_kde_meetroom_loc2_45cm():
    flist = [
            "data/audio/white-noise-true-20min.wav",
            "librispeech/test_wj/converted/Meeting-room/loc3-iph7-0p45m/convert/ATR-white-noise.wav",
            "librispeech/test_wj/converted/Meeting-room/loc3-iph7-0p45m/convert/clipon-white-noise.wav",
            "librispeech/test_wj/converted/Meeting-room/loc3-iph7-0p45m/convert/maono-white-noise.wav",
            "librispeech/test_wj/converted/Meeting-room/loc3-iph7-0p45m/convert/USB-white-noise.wav",
            "librispeech/test_wj/converted/Meeting-room/loc3-iph7-0p45m/convert/USBplug-white-noise.wav",
            "librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/ATR-silence.wav",
            "librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/clipon-silence.wav",
            "librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/maono-silence.wav",
            "librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/USB-silence.wav",
            "librispeech/test_wj/converted/Meeting-room/loc1-iph7/convert/USBplug-silence.wav"
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
