# PhyAug for Automated Speech Recognition (ASR)
This is PyTorch implementation of ASR case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentationfor Deep Sensing Model Transfer inCyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

This Repo is based on [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/README.md). **The credit goes to the corresponding authors**.

## Implementation details
We apply PhyAug on [DeepSpeech2](https://arxiv.org/pdf/1512.02595v1.pdf) ASR model. The model is pretrained on [Librispeech corpus](https://www.openslr.org/12). DeepSpeech2 is difficult to implement, please find the original [repository](https://github.com/SeanNaren/deepspeech.pytorch) for detailed implementation.

## Dataset
- Follow link to download the [Librispeech corpus](https://www.openslr.org/12) dataset.
- We use five microphones (M1,M2,M3,M4 and M5) to record the white noise and test dataset in a quiet meeting room. The dataset and pretrained model is available from this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/1IM0MD)
- To train/evaluate the model, download the above datasets and put dataset in `/data` folder and organize them following below hierarchy:
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
        /M1
            /1.wav
            /2.wav
                .
                .
                .
        /M2
        /M3
        /M4
        /M5
```
## Manifest files
A manifest file is a csv file that contains the path to the audio wave and corresponding transcript. We have include the manifest files in this repo. Please find them in `/manifest` folder.

## Installation
Install ctcdecode and etcd follows the instruction in original [repository](https://github.com/SeanNaren/deepspeech.pytorch).
Then install the python packages use `pip`:

```bash
pip install -r deepspeech_pytorch_requirement.txt
```
## Train model
To train model on librispeech training dataset, run below python code on command line. Due to complexity of model and large size of training dataset, the training of DeepSpeech2 is quite slow. Please find the pretrained model `librispeech_pretrained_v2.pth` in `/model` folder.

Our model is trained on workstation with 4x NVIDIA 11GB RTX2080Ti GPUs, you may change the `device-ids` and `batch-size` to suit your workstation.

To train a new model:
```bash
python -m multiproc train.py  --train-manifest manifest/libri_train_manifest.csv --val-manifest manifest/libri_val_manifest.csv --epochs 80 --num-workers 16 --cuda --device-ids 0,1,2,3 --learning-anneal 1.01 --batch-size 48 --no-sortaGrad --visdom  --opt-level O1 --loss-scale 1 --id libri --checkpoint --save-folder model/ --model-path model/your_model_name.pth
```
To continue training:
```bash
python -m multiproc train.py  --train-manifest manifest/libri_train_manifest.csv --val-manifest manifest/libri_val_manifest.csv --epochs 80 --num-workers 16 --cuda  --device-ids 0,1,2,3 --learning-anneal 1.01 --batch-size 48 --no-sortaGrad --visdom  --opt-level O1 --loss-scale 1 --id libri --checkpoint --save-folder model/ --model-path model/your_model_name.pth --continue-from model/librispeech_pretrained_v2.pth --finetune
```
## Evaluate model
The pretrained model WER on `/test_clean` is between 6% and 8%. To evaluate on original `/test_clean` dataset: 
```bash
python test.py --test-manifest manifest/libri_test_clean_manifest.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
```

When apply pretrained model on recorded microphone dataset, it's expected to see WER increase:
```bash
python test.py --test-manifest manifest/meeting-room/loc3-iph7-0p45m/atr_list.csv --lm-path model/3-gram.pruned.3e-7.arpa --decoder beam --alpha 1.97 --beta 4.36 --model-path model/librispeech_pretrained_v2.pth --lm-workers 8 --device-id 3 --num-workers 16 --cuda --half --beam-width 1024 ;
```
## PhyAug for DeepSpeech
We proposed a fast microphone profiling by playing back white noise and record use microphones. The white noise data is downloadable along with microphone data and placed in `\white_noise` folder. It contains original white noise data and corresponding microphone white noise. Each white noise data is ~5mins length.

To apply PhyAug for DeepSpeech, only original training dataset and microphone white noise are needed. The pretrained model `deepspeech_meetingroom_PhyAug.pth` is downloadable along with recorded dataset.

```bash
python -m multiproc train_PhyAug.py  --train-manifest manifest/libri_train_manifest.csv --val-manifest manifest/libri_val_manifest.csv --epochs 80 --num-workers 16 --cuda  --device-ids 0,1,2,3 --learning-anneal 1.01 --batch-size 48 --no-sortaGrad --visdom  --opt-level O1 --loss-scale 1 --id PhyAug_for_librispeech --checkpoint --save-folder model/ --model-path model/your_model_name.pth --continue-from model/librispeech_pretrained_v2.pth --finetune
```
## Hardware specification
The above models are pretrained on a workstation with following specifications:
- CPU: Intel Core i9-7900X 3.30GHz 13.75MB Cache 10C/20T
- GPU: 4x Zotac nVidia RTX2080Ti 11GB GDDR6 PCIe x16 GPU Card