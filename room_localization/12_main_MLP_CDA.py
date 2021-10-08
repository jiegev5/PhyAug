import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import utils
from datagen_CNN import load_spec,get_class_number
# import dataloader
from models import main_models


def train(opt, train_dataloader):
    model = main_models.MLP(in_feature=163,output_feature=cls_num)
    # summary(model,(1,12,48),batch_size=-1,device='cpu')
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    trainLoss, testLoss = [], []
    train_loss_min = np.inf
    for epoch in range(opt['n_epoches']):
        tloss = 0.0
        valloss = 0.0
        model.train()
        for data, labels in train_dataloader:
            # print(f'mnist dataset: mean: {data.mean()} max: {data.max()} min: {data.min()} std: {data.std()}')
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(data)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()
            # update train loss
            tloss += loss.item() * data.size(0)

        acc = eval(model, test_dataloader)
        tloss = tloss / len(train_dataloader.dataset)
        valloss = valloss / len(test_dataloader.dataset)
        trainLoss.append(tloss)
        testLoss.append(valloss)
        print("Epoch: %d/%d  train loss: %.3f test acc: %.3f" % (epoch + 1, opt['n_epoches'], tloss, acc))
        # save best model
        if tloss <= train_loss_min:
            print('Train loss decreased ({:.3f} --> {:.3f}).  Saving model ...'.format(
                train_loss_min,
                tloss))
            torch.save(model.state_dict(), './saved_models/model_mlp.pt')
            train_loss_min = tloss


def eval(model, data_dataloader):
    acc = 0
    model.eval()
    for data, labels in data_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = model(data)
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
        accuracy = round(acc / float(len(data_dataloader)), 3)
    return accuracy


def eval_plot_confusionMatrix(dataloader, device, plot=True):
    model = main_models.MLP(in_feature=163,output_feature=cls_num)
    # summary(model,(1,12,48),batch_size=-1,device='cpu')
    model.to(device)
    # save model
    # torch.save(model.state_dict(), 'saved_models/model_spec.pt')
    model.load_state_dict(torch.load('./saved_models/model_mlp.pt'));

    class_correct = [0] * cls_num
    class_total = [0] * cls_num
    conf_matrix = np.zeros((cls_num, cls_num))
    model.eval()

    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = model(data)
        # acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

        _, pred = torch.max(y_test_pred, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())

        # Update confusion matrix
        for i in range(labels.size(0)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

            # Update confusion matrix
            conf_matrix[label][pred.data[i]] += 1

    # print accuracy for each class
    for i in range(cls_num):
        if class_total[i] > 0:
            print('Test Accuracy of %3s: %2d%% (%2d/%2d)' % (
                i, 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %3s: N/A (no training examples)' % (i))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    if plot:
        plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(conf_matrix.astype(int), annot=True, fmt = "d", square=True, vmax=20)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()


def inf_for_uploaded_file(PCM_file):
    # load model
    model = main_models.CNN_SPEC(num_classes=cls_num)
    # summary(model,(1,12,48),batch_size=-1,device='cpu')
    model.to(device)
    model.load_state_dict(torch.load('./saved_models/model_spec.pt'))
    data = utils.convert_PCM_to_CSV(PCM_file, 'test')
    # mean = data.mean()
    # std = data.std()
    # data = (data - mean) / std  # normalize data
    data = np.array(data.reshape(-1, 12, 48), dtype=np.float32)
    data = torch.tensor(data.reshape(-1, 1, 12, 48), dtype=torch.float32)  # convert to correct shape
    data = data.to(device)
    y_test_pred = model(data)
    # print(y_test_pred.cpu())
    # print(torch.argmax(y_test_pred,dim=1))
    # pred = torch.argmax(y_test_pred, 1) 
    # print(pred) 
    _, pred = torch.max(y_test_pred, 1)
    pred = np.squeeze(pred.numpy()) if device == "cpu" else np.squeeze(pred.cpu().numpy())
    value, count = np.unique(pred, return_counts=True)
    # print(value)
    # print(count)
    # print(f'predicted labels is: {label_index[value[np.argmax(count)]]}')
    print("predicted class: ",value[np.argmax(count)])
    return value[np.argmax(count)]

def get_train_data_class_num(path_to_train_data):
    labels = []
    training_data_files = glob.glob(path_to_train_data)
    for f in training_data_files:
        lab = f.split('_')[1]
        labels.append(lab)
    cls_num = len(np.unique(labels))
    return cls_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)

    opt = vars(parser.parse_args())

    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)

    # class to index mapping
    # s_mic = 'motorz'
    # t_mic = 'pixel4'
    # exp = 'exp_ABS_N4_4_rooms_{}_to_{}'.format(s_mic,t_mic)

    exp = 'exp_ABS_N4_20_rooms_{}_augmented_kde'.format('motorz')
    cls_num = get_class_number(exp)
    # cls_num = get_train_data_class_num(os.path.join('./data/pixel4/train', "*.csv"))
    print("class number: ",cls_num)

    retrain = True  # if retrain model
    if retrain:
        # PCM_file = 'xxx'
        # utils.convert_PCM_to_CSV(PCM_file, opt='train')
        model_dir = './saved_models'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        train_dataloader, test_dataloader = load_spec(opt=exp,batch_size=opt['batch_size'])
        train(opt, train_dataloader)

    exp_list = [
        'exp_ABS_N4_20_rooms_pixel4',
        'exp_ABS_N4_20_rooms_motorz',
        'exp_ABS_N4_20_rooms_galaxys7',
    ]
    for exp in exp_list:
        train_dataloader, test_dataloader = load_spec(opt=exp,batch_size=opt['batch_size'])
        eval_plot_confusionMatrix(test_dataloader,device,plot=False)

    # inf_for_uploaded_file('/data2/wenjie/echo_localization/DATA_FOLDER/echo_data/pcm/NCS_whole_lab/test_37.pcm')
