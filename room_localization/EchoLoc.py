import glob
import os

import numpy as np
import torch

import utils
# import dataloader
from .models import main_models


class EchoLoc:
    def __init__(self, model_weights_path, training_data_dir):
        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
        torch.manual_seed(1)
        if use_cuda:
            torch.cuda.manual_seed(1)
        print("[EchoLoc] Initialize the torch, device:", self.device)
        self.training_dir = os.path.join(training_data_dir, "*.csv")
        self.training_label_index = self.get_train_data_label_index(self.training_dir)
        class_number = self.get_train_data_class_num(self.training_dir)
        print("[EchoLoc] Class number:", class_number)
        self.model = main_models.CNN_SPEC(num_classes=class_number)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.to(self.device)
        print("[EchoLoc] Load model done")

    def get_train_data_label_index(self, path_to_train_data):
        training_data_files = glob.glob(path_to_train_data)
        label_index = utils.get_label_index_dict(training_data_files)
        return label_index

    def get_train_data_class_num(self, path_to_train_data):
        training_data_files = glob.glob(path_to_train_data)
        cls_num = len(training_data_files)
        return cls_num

    def inf_for_uploaded_file(self, pcm_file):
        # load model
        data = utils.convert_PCM_to_CSV(pcm_file, 'test')
        data = (data - 0.5) / 0.5  # normalize data
        data = np.array(data.reshape(-1, 12, 48), dtype=np.float32)
        data = torch.tensor(data.reshape(-1, 1, 12, 48), dtype=torch.float32)  # convert to correct shape
        data = data.to(self.device)
        y_test_pred = self.model(data)
        _, pred = torch.max(y_test_pred, 1)
        pred = np.squeeze(pred.numpy()) if self.device == "cpu" else np.squeeze(pred.cpu().numpy())
        value, count = np.unique(pred, return_counts=True)
        print('[EchoLoc] predicted labels is:{}'.format(self.training_label_index[value[np.argmax(count)]]))
        return pred


if __name__ == '__main__':
    testEchoLoc = EchoLoc(model_weights_path='../data/saved_models/model_spec.pt'
                          , training_data_dir='../data/train'
                          )
    test_file_path = '/data2/wenjie/echo_localization/DATA_FOLDER/echo_data/pcm/NCS_demo_room/test_10.pcm'
    testEchoLoc.inf_for_uploaded_file(test_file_path)
