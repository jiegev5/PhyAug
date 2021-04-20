# PhyAug for Automated Speech Recognition (ASR)
This is PyTorch implementation of ASR case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentation for Deep Sensing Model Transfer in Cyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

This Repo is based on [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/README.md). **The credit goes to the corresponding authors**.

## Model implementation
We apply PhyAug on [DeepSpeech2](https://arxiv.org/pdf/1512.02595v1.pdf) ASR model. The model is pretrained on [Librispeech corpus](https://www.openslr.org/12). Please find the original [repository](https://github.com/SeanNaren/deepspeech.pytorch) for detailed implementation.

## Dataset
- Follow the link to download the [Librispeech corpus](https://www.openslr.org/12) dataset.
- We use five microphones (M1,M2,M3,M4 and M5) to record the white noise and Librispeech dataset in a quiet meeting room. The dataset and pretrained models are available from this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/A6SC66). The dataset consists of white noise data and recorded Librispeech clean test data. Each recorded white noise data is 5 minutes long. The recorded Librispeech clean test data consists of 2619 test audios, totaling 5.5 hours long for each microphone.
- To train/evaluate the model, download the above dataset and place in `/data` folder, organize them by following below hierarchy:
```
/librispeech
    /train
        /wav
        /txt
    /eval
        /wav
        /txt
    /test_clean
        /wav
        /txt
    /recorded_dataset
        /white_noise
            M1_white_noise.wav
            M2_white_noise.wav
                    .
                    .
        /M1
            1.wav
            2.wav
                .
                .
        /M2
        /M3
        /M4
        /M5
```
## Manifest files
A manifest file is a `.csv` file that contains the path to the audio wave and the corresponding transcript. It's used for model training and evaluation. We upload the manifest files in this repo, please find them in `/manifest` folder. An example of manifest looks like below (*Please double check the paths in each manifest file if they match your actual files' path*)
```bash
librispeech/test_clean/wav/2830-3980-0026.wav,librispeech/test_clean/txt/2830-3980-0026.txt
...
```

## Installation
The following installation is tested on Ubuntu 18.04 with Python 3.6.9 and Ubuntu 20.04 with Python 3.8.5. We install required python packages using `pip`. 

Install [Pytorch](https://pytorch.org/get-started/locally/).

Then, run the following commands in a terminal to install the original [repository](https://github.com/SeanNaren/deepspeech.pytorch).

```bash
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..
git clone https://github.com/SeanNaren/deepspeech.pytorch.git
pip install -r requirements.txt
sudo apt-get install etcd 
```


Install [Apex](https://github.com/NVIDIA/apex#linux) from NVIDIA.

```bash
sudo apt-get install swig
pip install -r deepspeech_pytorch_requirement.txt
```
## Train model
To train model on librispeech training dataset, run below python code on command line. Due to the complexity of model and extrememly large size of training dataset, the training of DeepSpeech2 is quite slow. The pretrained model `librispeech_pretrained_v2.pth` is downloadable along with our uploaded dataset, please download it and place in `/model` folder.

To train a new model (assume multiple GPUs are available):
```bash
python -m multiproc train.py  --train-manifest manifest/libri_train_manifest.csv --val-manifest manifest/libri_val_manifest.csv --epochs 80 --num-workers 16 --cuda --device-ids 0,1,2,3 --learning-anneal 1.01 --batch-size 48 --no-sortaGrad --visdom  --opt-level O1 --loss-scale 1 --id libri --checkpoint --save-folder model/ --model-path model/your_model_name.pth
```
To continue training from a pretrained model:
```bash
python -m multiproc train.py  --train-manifest manifest/libri_train_manifest.csv --val-manifest manifest/libri_val_manifest.csv --epochs 80 --num-workers 16 --cuda  --device-ids 0,1,2,3 --learning-anneal 1.01 --batch-size 48 --no-sortaGrad --visdom  --opt-level O1 --loss-scale 1 --id libri --checkpoint --save-folder model/ --model-path model/your_model_name.pth --continue-from model/librispeech_pretrained_v2.pth --finetune
```
The model will be saved in `/model` folder, you may give a new model name by replacing `model/your_model_name.pth`

Note that our model is evaluated on workstation with 4x NVIDIA 11GB RTX2080Ti GPUs, you may change the `device-ids` and `batch-size` to suit your workstation. Model training time is around 48 hours on our workstation.

## Evaluate model
The word error rate (WER) of pretrained model on Librispeech clean dataset (`/test_clean`) is between 6% and 8%. To obtain the same result, run below command: 
```bash
python test.py --test-manifest manifest/libri_test_clean_manifest.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
```

When applying pretrained model on recorded microphone dataset, it's expected to see WER increases:
```bash
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/atr_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
```
## PhyAug for DeepSpeech
A microphone can be characterized by its frequency response. We proposed a fast microphone profiling by playing back and recording white noise using microphones, then use white noise to estimate a microphone's frequency response curve (FRC). 

The white noise data is downloadable along with microphones' dataset, they are placed in `\white_noise` folder. It contains original white noise data and the corresponding microphone white noise. Each white noise data is around 5 minutes long.

To apply PhyAug for DeepSpeech, only original training dataset and microphone white noise are needed. Run below command to use microphones' FRC to retrain model. 

```bash
python -m multiproc train_PhyAug.py  --train-manifest manifest/libri_train_manifest.csv --val-manifest manifest/libri_val_manifest.csv --epochs 80 --num-workers 16 --cuda  --device-ids 0,1,2,3 --learning-anneal 1.01 --batch-size 48 --no-sortaGrad --visdom  --opt-level O1 --loss-scale 1 --id PhyAug_for_librispeech --checkpoint --save-folder model/ --model-path model/your_model_name.pth --continue-from model/librispeech_pretrained_v2.pth --finetune
```
The pretrained model `deepspeech_meetingroom_PhyAug.pth` is downloadable in provided link.

To evaluate on microphone dataset use retrained model:
```bash
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/atr_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/deepspeech_meetingroom_PhyAug.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
```

The result corresponds to Figure 9 in our paper.

[**Back to front page**](../README.md)
