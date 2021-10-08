import os
import csv
import numpy as np
from scipy import signal
# %matplotlib inline
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def findStart(data, threshold):
    """
    find the start of audio
    """
    start = 0
    # find the first point of signal
    for index, amp in enumerate(data):
        if amp >= threshold:
            start = index
            break
    return start


def extract_echo_spectrum12x48(data, start):
    chirp_len = 500
    # start = utils.findStart(data,chirp_len)

    direct_path = data[start:start + chirp_len]  # direct audio
    # 2900 - 72
    # 2650 - 64
    echo = data[start + int(chirp_len * 1.1):start + 2920]  # 2900 - 72
    fs = 44100
    # high pass filter
    sos = signal.butter(5, 10000, 'hp', fs=44100, output='sos')
    filtered = signal.sosfilt(sos, echo)
    # plt.plot(t,data)

    f0 = 15000
    f1 = 20000
    nfft = 96
    noverlap = (nfft / 2)
    t = len(filtered) / float(fs)
    ff, tt, Sxx = spectrogram(filtered, fs=fs, noverlap=noverlap, nperseg=nfft, \
                              mode='magnitude')
    # plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='hot')
    segment = (fs / 2) / noverlap
    lower_f = int(f0 / segment) - 1
    upper_f = int(f1 / segment)  # given some bandwidth
    f = ff[lower_f:upper_f]
    S = Sxx[lower_f:upper_f, :]
    # zscore normalization
    # mean = np.mean(S)
    # std = np.std(S)
    # S = (S - mean)/std
    S = S / np.abs(S).max()

    return f, tt, S


def get_file_pref(f):
    ftemp = f.split(os.sep)[-1]  # get file name
    return ftemp.split('.')[0]


def convert_PCM_to_CSV(pcm_file, opt):
    if opt == 'train':
        save_folder = 'data/train'
    if opt == 'test':
        save_folder = 'data/test'
    feature = np.array([])
    pref = get_file_pref(pcm_file)  # get file prefix
    data = np.memmap(pcm_file, dtype='int16', mode='r')
    while len(data) > 5000:
        # for j in range(samples): # how many data samples
        start = findStart(data, 2500)
        # print("length of data: ",len(data))
        f, t, Sxx = extract_echo_spectrum12x48(data, start)
        # print(f'file: {pref},spec shape: {Sxx.shape}, max: {Sxx.max()},min: {Sxx.min()}')
        new_start = start + 3500
        data = data[new_start:]
        feature = np.vstack([feature, Sxx.flatten()]) if feature.size else Sxx.flatten()
    np.savetxt(os.path.join(save_folder, pref + '.csv'), feature, delimiter=",")
    return feature


def get_label_index_dict(flist):
    label_index = {}
    for i in range(len(flist)):
        f = get_file_pref(flist[i])
        label_index[i] = f
        i += 1
    return label_index

def findStartoftSound(data, num):
    f = []  # start of each peak
    # plt.plot(data)
    # plt.show()
    for i in range(0, num):
        if i == 0:
            f.append(0)
        else:
            for j in range(-500, 500):
                if data[i * 4600 + j] >= 2000:
                    f.append(i * 4600 + j)
                    break

    return f

def findStartoftSoundV1(data):
    f = []  # start of each peak
    temp_d = data.copy()
    i = 0
    while len(temp_d) > 5000:
        if i == 0:
            f.append(0)
            i += 1
        else:
            for j in range(-200, 200):
                if data[i * 4600 + j] >= 5000:
                    f.append(i * 4600 + j)
                    temp_d = data[i * 4600 + j:]
                    i += 1
                    break
    print(f'{i} echoes found')
    return f


def _get_sep(path):
    if isinstance(path, bytes):
        return b'/'
    else:
        return '/'

def basename(p):
    """Returns the final component of a pathname"""
    p = os.fspath(p)
    sep = _get_sep(p)
    i = p.rfind(sep) + 1
    return p[i:]

def getFilename(filename):
    plot_name = basename(filename)
    plot_name = os.path.splitext(plot_name)[0]
    plot_name = plot_name + ".csv"

    return plot_name

def jointData(data, s_begin, num):
    ndata = []
    for i in range(0, num):
        tmp_data = []
        for j in range(0, 4300):
            tmp_data.append(data[s_begin[i] + 100 + 100 + j]) # extra 100 for safe guarding
        ndata.append(tmp_data)
    return ndata

def jointDirectData(data, s_begin, num):
    ndata = []
    for i in range(0, num):
        start = s_begin[i]
        ndata.extend(data[start:start+100])
    return ndata


def calLongPSDsof3800Points(ndata):
    power_data, freq_psd = plt.psd(ndata, Fs=fs, NFFT=7600, visible=False)

    plt.xlim(19486, 20341)
    plt.ylim(-1, 20)
    plt.plot(freq_psd, power_data, marker='+')

    return power_data

def calSpectrogram(data, fs, start, end):
    data = np.array(data)
    # fig, ax = plt.subplots()
    # cmap = plt.get_cmap('viridis')
    # vmin = 0
    # vmax = 50
    # cmap.set_under(color='k', alpha=None)
    NFFT = 256
    freq, t, pxx = spectrogram(data, fs=fs, noverlap=NFFT/2, nperseg=NFFT, \
                              mode='magnitude')
    # pxx, freq, t, cax = ax.specgram(data / (NFFT / 2), Fs=fs, mode='magnitude',
    #                                 NFFT=NFFT, noverlap=NFFT / 2,
    #                                 vmin=vmin, vmax=vmax, cmap=cmap)
    # fig.colorbar(cax)
    # plt.show()
    # print(data.shape,pxx.shape,pxx.max(),pxx.mean())
    # pxx_1d = []

    # for i in range(start, end):
    #     for j in range(len(pxx[i])):
    #         pxx_1d.append(pxx[i][j])

    return pxx[start:end,:], freq, t

fs = 44100  # sampling rate
starting_amplitude = 2500
segment_length = 3800
end_of_inertia_part = 600
total_segment_length = 179400
segment_interval = 188220

def PCMToCSV_train(file_name, output_file_dir, mode, ndata = 2, tf = None, tf_mode = 'A'):
    counter = 0
    output_file = getFilename(file_name)
    output_file = os.path.join(output_file_dir,output_file)
    # print(output_file)
    # turn PCM to memmap
    data = np.memmap(file_name, dtype='h', mode='r')
    sos = signal.butter(5, 15000, 'hp', fs=fs, output='sos')
    data = signal.sosfilt(sos, data) # filtered data

    # cut launch part of data
    for sample_index, signal_amplitude in enumerate(data):
        if signal_amplitude >= starting_amplitude:
            start_of_data = sample_index
            break
    data = data[start_of_data:]

    # segment data

    # f: find the start of every peak within 50 peaks
    f = findStartoftSound(data,ndata)
    # print(f'start is {f}, lenth: {len(f)}')
    # ndata: the 3800 points array,ndata.size()=50
    ndata = jointData(data, f, ndata)
    print(len(ndata))
    for i in range(len(ndata)):
        counter += 1
        if mode == "3800LongPSDs":
            power_data = calLongPSDsof3800Points(ndata[i])
            csvfile = open(output_file, 'a')
            fw = csv.writer(csvfile, quoting=csv.QUOTE_NONE, lineterminator='\n')
            fw.writerow(power_data[3358:3505])
            csvfile.close()
        if mode == "Spectrogram":
            pxx, freq, t = calSpectrogram(ndata[i], fs, 114, 119)
            spec = np.array(pxx)
            if tf is not None:
                if tf_mode == 'A':
                    max = spec.max()
                    min = spec.min()
                    spec = (spec - min)/max()
                    spec = np.multiply(spec,tf)
                    # spec = spec.flatten()
                if tf_mode == 'B':
                    spec = np.multiply(spec,tf)
            # scale
            max = spec.max()
            # min = spec.min()
            spec = spec / max  # normalize data
            spec = spec.flatten()
            # fig, ax = plt.subplots(figsize=(5, 5))            
            # plt.imshow(spec.reshape(5,32),cmap='gray_r',aspect=4)
            # plt.tight_layout()
            # plt.show()
            csvfile = open(output_file, 'a')
            fw = csv.writer(csvfile, quoting=csv.QUOTE_NONE, lineterminator='\n')
            fw.writerow(spec)
            plt.close('all')
            csvfile.close()
    return output_file

def PCMToNPY_CYCLEGAN(file_name, 
                      output_file_dir, 
                      ndata = 1000, 
                      start = 0):
    counter = start
    output_file = getFilename(file_name)
    output_file = os.path.join(output_file_dir,output_file)
    # print(output_file)
    # turn PCM to memmap
    data = np.memmap(file_name, dtype='h', mode='r')
    sos = signal.butter(5, 15000, 'hp', fs=fs, output='sos')
    data = signal.sosfilt(sos, data) # filtered data

    # cut launch part of data
    for sample_index, signal_amplitude in enumerate(data):
        if signal_amplitude >= starting_amplitude:
            start_of_data = sample_index
            break
    data = data[start_of_data:]

    # segment data

    # f: find the start of every peak within 50 peaks
    f = findStartoftSound(data,ndata)
    # print(f'start is {f}, lenth: {len(f)}')
    # ndata: the 3800 points array,ndata.size()=50
    ndata = jointData(data, f, ndata)
    print(len(ndata))
    for i in range(len(ndata)):
        pxx, freq, _ = calSpectrogram(ndata[i], fs, 112, 120)
        spec = np.array(pxx)
        # scale
        max = spec.max()
        mean = spec.mean()
        min = spec.min()
        std = spec.std()
        # min = spec.min()
        spec = (spec - mean)/(max - min)
        fname = os.path.join(output_file_dir,str(counter)+'.npy')
        np.save(fname,spec.reshape(1,8,32))
        counter += 1
    return 

def PCM2DIRECTSPEC(file_name, ndata = 400):
    # turn PCM to memmap
    data = np.memmap(file_name, dtype='h', mode='r')
    sos = signal.butter(5, 15000, 'hp', fs=fs, output='sos')
    data = signal.sosfilt(sos, data) # filtered data

    # cut launch part of data
    for sample_index, signal_amplitude in enumerate(data):
        if signal_amplitude >= starting_amplitude:
            start_of_data = sample_index
            break
    data = data[start_of_data:]
    # print(len(data))
    # segment data

    # f: find the start of every peak within 50 peaks
    f = findStartoftSound(data,ndata)
    # print(f)
    # ndata: the 3800 points array,ndata.size()=50
    ndata = jointDirectData(data, f, ndata)
    # print(len(ndata))
    pxx, freq, t = calSpectrogram(ndata, fs, 114, 119)
    spec = np.array(pxx)
    # print(spec.shape)
    # fig, ax = plt.subplots(figsize=(5, 5))            
    # plt.imshow(spec.reshape(5,-1),cmap='gray_r',aspect=4)
    # plt.tight_layout()
    # plt.show()

    return spec.reshape(5,-1)

def PCMToEchoSpec(file_name, ndata = 2):
    # print(output_file)
    # turn PCM to memmap
    data = np.memmap(file_name, dtype='h', mode='r')
    sos = signal.butter(5, 15000, 'hp', fs=fs, output='sos')
    data = signal.sosfilt(sos, data) # filtered data

    # cut launch part of data
    for sample_index, signal_amplitude in enumerate(data):
        if signal_amplitude >= starting_amplitude:
            start_of_data = sample_index
            break
    data = data[start_of_data:]

    # segment data

    # f: find the start of every peak within 50 peaks
    f = findStartoftSound(data,ndata)
    # print(f'start is {f}, lenth: {len(f)}')
    # ndata: the 3800 points array,ndata.size()=50
    ndata = jointData(data, f, ndata)
    print(len(ndata))
    spec = np.array([])
    for i in range(len(ndata)):
        pxx, _, _ = calSpectrogram(ndata[i], fs, 114, 119)
        tspec = np.array(pxx)
        tspec = tspec/tspec.max()
        # max = tspec.max()
        # min = tspec.min()
        # tspec = (tspec - min) / max  # normalize data
        spec = np.vstack([spec,tspec.reshape(1,5,-1)]) if spec.size else tspec.reshape(1,5,-1)
    return spec

def detect_and_mkdir(dir_path):
    if False is os.path.isdir(dir_path):
        os.mkdir(dir_path)

def obtain_tf(fA,fB,mode='A'):
    if mode == 'A':
        '''
        use direct path to transfer
        '''
        spec_A = PCM2DIRECTSPEC(file_name=fA,ndata=300)
        spec_B = PCM2DIRECTSPEC(file_name=fB,ndata=300)
        spec_A = spec_A/spec_A.max()
        spec_B = spec_B/spec_B.max()
        AoverB = np.mean(spec_A,axis=1)/np.mean(spec_B,axis=1)
    if mode == 'B':
        '''
        use echo to transfer
        '''
        spec_A = PCMToEchoSpec(file_name=fA,ndata=300)
        spec_B = PCMToEchoSpec(file_name=fB,ndata=300)
        tf = np.divide(spec_A,spec_B)
        tf = np.mean(tf,axis=0) # aggregate along 0-th axis
        # print(tf.shape)
        # print(tf.max(),tf.min())
        # print(spec_A[0][0])
        # print(spec_B[0][0])
        # print(tf[0][0])

    return tf

def read_csv(file):
    return np.genfromtxt(file, delimiter=',')

def read_pcm(file):
    data = np.memmap(file,dtype='int16',mode='r')
    t = np.linspace(0, len(data) / fs, num=len(data))
    return data,t

class extract_spectrogram(object):
    def __init__(self, 
                 plot=False): 
        self.fs = 44100 # sampling rate
        self.plot = plot

    def get_tf(self,device_A,device_B):
        '''
        return tf = A./B
        '''
        _,_,fA = self.extract_whitenoise_spec(device_A)
        _,_,fB = self.extract_whitenoise_spec(device_B)

        # newly added
        fA = self.sort_spec(fA)
        fB = self.sort_spec(fB)

        # aggregate feature
        fA = np.mean(fA,axis=1)
        fB = np.mean(fB,axis=1)
        # print(fA.shape,fB.shape)
        return np.divide(fA,fB)

    def get_tf_kde(self,device_A,device_B,source_device):
        fA = self.read_pcm_extract_spec_feature(device_A)
        fB = self.read_pcm_extract_spec_feature(device_B)
        fS = self.read_pcm_extract_spec_feature(source_device)
        # print(fB.shape,fB.max(),fB.min())
        # print(fS.shape,fS.max(),fS.min())
        min_lenth = min(fA.shape[0],fB.shape[0],fS.shape[0])
        tf_A = np.divide(fA[:min_lenth],fS[:min_lenth])
        tf_B = np.divide(fB[:min_lenth],fS[:min_lenth])
        # print(tf_A.shape,tf_A.max(),tf_A.mean())
        kde_A = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_A)
        kde_B = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(tf_B)

        # sample_A = kde_A.sample(1)
        # print(sample_A.max(),sample_A.min())
        # sample_A1 = fA[1]
        # plt.plot(sample_A1.reshape(-1,))
        # plt.show()
        # print(sample_A.shape,sample_A)
        return [kde_A,kde_B]

    def read_pcm_extract_spec_feature(self,device):
        """
        extract all features given a device name
        """
        root = './data/pcm/Batphone/{}/white_noise'.format(device)
        file = os.path.join(root,'test_20_.pcm') #   test_3_background
        data = np.memmap(file,dtype='int16',mode='r')
        splited_array = [data[i:i + self.fs] for i in range(0, len(data), self.fs)]
        inf_count = 0
        samples = 0
        feature = np.array([])

        for j in range(len(splited_array)-1):
            # print(splited_array[j])
            feat = self.extract_single_feature(splited_array[j])
            if feat.min() <= 0: # check infinity number
                inf_count += 1
                continue
            samples += 1
            feature = np.vstack(
                [feature, feat]) if feature.size else feat

        return feature

    def extract_single_feature(self, data):
        nfft = 1024
        noverlap = nfft/2
        s = 0
        ff, tt, Sxx = spectrogram(
            data, fs=fs, noverlap=noverlap, nperseg=nfft, window='hamming', mode='psd')
        segment = (fs/2)/noverlap
        lower_f = 0
        upper_f = int(7000/segment)+1  # given some bandwidth
        f = ff[lower_f:upper_f]
        S = Sxx[lower_f:upper_f, :]
        # sort each row
        S = np.sort(S, axis=1)
        # did not take log here
        feature = S[:, 4]
        # print("feature: ",feature.max(),feature.min()) # the 5th column
        # print(f,S.shape)
        return feature.reshape(1,-1)

    def sort_spec(self,spec):
        spec = np.sort(spec, axis=1) # sort along y
        col = spec.shape[1] # how many columns
        s = int(col*0.01) # 0.01
        e = int(col*0.05) # 0.05
        spec = spec[:,s:e] # only select 5% - 20%
        # spec = np.mean(spec,axis=1)
        return spec


    def extract_whitenoise_spec(self,device):
        '''
        return averaged PSD in each frequency bin
        '''
        nfft = 1024
        noverlap = nfft/2
        fs = 44100
        root = '../data/pcm/Batphone/{}/white_noise'.format(device)
        file = os.path.join(root,'test_20_.pcm') #   test_3_background
        data = np.memmap(file,dtype='int16',mode='r')
        s = 441000
        length = fs*100
        plot_data = data[s:length+s]
        ff, tt, Sxx = spectrogram(
            plot_data, fs=fs, noverlap=noverlap, nperseg=nfft, window='hamming', mode='psd')
        segment = (fs/2)/noverlap
        lower_f = 0
        upper_f = int(7000/segment)+1  # given some bandwidth
        f = ff[lower_f:upper_f]
        S = Sxx[lower_f:upper_f, :]
        if self.plot:
            fig, axs = plt.subplots(2, 1)
            axs[0].plot(np.arange(len(plot_data)) / float(fs), plot_data)
            axs[0].set_title("recorded audio")
            axs[1].pcolormesh(tt, f, S, cmap='hot')
            axs[1].set_title("Spectrogram")
            plt.tight_layout()
            plt.show()

        # get the mean value along y axis
        # feature = np.mean(S,axis=1)
        return tt,f,S

    def extract_background_spec(self,device):
        '''
        return averaged PSD in each frequency bin
        '''
        nfft = 1024
        noverlap = nfft/2
        fs = 44100
        root = '../data/pcm/Batphone/{}/white_noise'.format(device)
        file = os.path.join(root,'test_20_.pcm') #   test_3_background
        data = np.memmap(file,dtype='int16',mode='r')
        s = 44100
        length = fs
        plot_data = data[s:length+s]
        ff, tt, Sxx = spectrogram(
            plot_data, fs=fs, noverlap=noverlap, nperseg=nfft, window='hamming', mode='psd')
        segment = (fs/2)/noverlap
        lower_f = 0
        upper_f = int(500/segment)+1  # given some bandwidth
        f = ff[lower_f:upper_f]
        S = Sxx[lower_f:upper_f, :]
        if self.plot:
            fig, axs = plt.subplots(2, 1)
            axs[0].plot(np.arange(len(plot_data)) / float(fs), plot_data)
            axs[0].set_title("recorded audio")
            axs[1].pcolormesh(tt, f, S, cmap='magma')
            axs[1].set_title("Spectrogram")
            plt.tight_layout()
            plt.show()

        # get the mean value along y axis
        # feature = np.mean(S,axis=1)
        return tt,f,S

    def plot_spec_3_devices(self,deviceA,deviceB,deviceC):
        tt,f,SA = self.extract_background_spec(deviceA)
        tt,f,SB = self.extract_background_spec(deviceB)
        tt,f,SC = self.extract_background_spec(deviceC)
        
        fig, axs = plt.subplots(3, 1)
        axs[0].pcolormesh(tt, f, SA, cmap='magma')
        axs[0].set_title("Pixel 4")
        axs[1].pcolormesh(tt, f, SB, cmap='magma')
        axs[1].set_title("Galaxy s7")
        axs[2].pcolormesh(tt, f, SC, cmap='magma')
        axs[2].set_title("Motor Z")
        plt.tight_layout()
        plt.show()

        # get the mean value along y axis
        # feature = np.mean(S,axis=1)
        return 