# k-means clustering
import sys
import time
import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from glob import glob
import sys
import pickle


def extract_ABS_feature(data, tf = None, plot=False):
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
    if tf is not None:
        feature = S[:, 4].reshape(-1,)
        assert len(tf) == len(feature)
        feature_t = np.multiply(feature,tf)
        feature_t = np.log(feature_t)
        feature = np.vstack([np.log(feature),feature_t])
        # print(feature.shape)
        # sys.exit()
        return feature_t.reshape(1,-1)
    else:
        feature = np.log(S[:, 4].reshape(1,-1))
        return feature

    # print("feature: ",feature.max(),feature.min()) # the 5th column
    # # print(f,S.shape)
    # if plot:
    #     fig, axs = plt.subplots(2, 1)
    #     axs[0].plot(np.arange(len(data)) / float(fs), data)
    #     axs[0].set_title("recorded audio")
    #     axs[1].pcolormesh(tt, f, S, cmap='hot')
    #     axs[1].set_title("Spectrogram")
    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    s = time.time()
    fs = 44100
    target_phone = 'pixel4'
    source_phone = 'motorz'
    root = './data/pcm/Batphone/{}'.format(source_phone)
    option = ['train', 'test']
    train_index = np.arange(0, 20)  # how many locations
    test_index = train_index
    # fpost = 'exp_ABS_N4_{}_spec_4_rooms'.format(phone)
    fpost = 'exp_ABS_N4_20_rooms_{}_to_{}'.format(source_phone,target_phone)
    label_index = np.arange(len(train_index))
    feature_extractor = utils.extract_spectrogram()
    # get transfer function
    tf = feature_extractor.get_tf(target_phone,source_phone)

    for opt in option:
        feature = np.array([])
        labels = np.array([])
        inf_count = 0
        for i in range(len(train_index)):
            if opt == 'train':
                loc = str(train_index[i])  # convert to string
            else:
                loc = str(test_index[i])
            fname = '{}_{}_*.pcm'.format(opt, loc)
            files = glob(os.path.join(root, fname))
            print(f'class: {loc} {len(files)} detected')
            samples = 0
            for f in files:
                data = np.memmap(f, dtype='int16', mode='r')
                # print(data.shape)
                splited_array = [data[i:i + fs]
                                 for i in range(0, len(data), fs)]
                # print(splited_array)
                for j in range(len(splited_array)-1):
                    # print(splited_array[j])
                    feat = extract_ABS_feature(splited_array[j], tf=tf, plot=False)
                    if feat.min() == np.NINF: # check infinity number
                        inf_count += 1
                        continue
                    # feat = np.array(feat).reshape(1, -1)
                    samples += feat.shape[0]
                    feature = np.vstack(
                        [feature, feat]) if feature.size else feat
            cls_labels = np.zeros(samples) + train_index[i]
            labels = np.concatenate(
                (labels, cls_labels)) if labels.shape else cls_labels

        print(f"final feature: {feature.shape}, label shape: {labels.shape}, inf count: {inf_count}")
        folder = './data/spec'
        # dump data to pickle files
        save_name = os.path.join(folder, '{}_{}.p'.format(opt, fpost))
        dict_ = {'features': feature, 'labels': labels}
        outfile = open(save_name, 'wb')
        pickle.dump(dict_, outfile)
        outfile.close()
    e = time.time()
    print(f'Script running time: {round((e - s)/60,2)} minutes')
    print("__End of Script__")
