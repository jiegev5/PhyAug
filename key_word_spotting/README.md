# PhyAug for Keyword Spotting

This is PyTorch implementation of keyword spotting (KWS) case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentation for Deep Sensing Model Transfer in Cyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

This Repo is based on [Honk](https://github.com/castorini/honk): A PyTorch reimplementation of Google's TensorFlow convolutional neural networks for keyword spotting. 
**The credit goes to the corresponding authors**.

## Implementation details
This case study applies Convolutional neural network for keyword spotting. Specifically, we apply "cnn-trad-pool2" model to discriminate 12 commands: "yes,no,up,down,left,right,on,off,stop,go,silence and unknown".

## dataset
- The original data set is downloadable from this [link](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html).
- We use five microphones (M1,M2,M3,M4 and M5) to record the white noise and speech data in a quiet meeting room. The recored dataset is available from this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/1IM0MD). This dataset consists of 5 microphones’ white noise data and recorded google speech commands data collected in a quiet meeting room. Each white noise data is around 5 minutes long. The recorded speech command data comprises 12 commands' test data, totaling 4074 audio utterances, and 68 minutes long.
- To train/evaluate the model, download the above datasets and put in `/data` folder, organize them following this hierarchy:
```
/data
    /original_dataset
        /class1
        /class2
            .
            .
    /recorded_dataset
        /white_noise
        /M1
            /class1
            /class2
                .
                .
                .
        /M2
        /M3
        /M4
        /M5
```
## Installation
Set up the virtual environment and use pip to install packages. Noted that we evaluate under python3.6.9:

```bash
pip install -r PhyAug_kws_requirement.txt
```
## Train model
Train/evaluate script in place in `/utils` folder, please `cd` to this folder and follow below instruction for model training/evaluation.

To train on original dataset:
- Set `use_tf = False` at line 14 in `train_PhyAug.py`, this follows standard procedure to train a CNN without using PhyAug.
- You need to modify the directory location at line 554 in `model.py`, point to the location of your speech dataset: `config["data_folder"] = "your/speech_dataset/location"` 

```bash
python train_PhyAug.py --type train
```
The model training time on our workstation is ~2hours for 40 epoches.

## Evaluate model
The model accuracy on original test dataset is 90%. To evaluate model on collected microphone dataset, set `config["data_folder"] = "../data/recorded_dataset/M1"` in `model.py`, point to the location of microphone dataset. Noted that you need to do this for each microphone dataset. It's expected to see accuracy drop if we test pretained model on collected microphone dataset. We provide our pretrained model `model_12cmd_original.pt` in `/model` folder.

```bash
python evaluate_model.py --type eval --input_file ../model/model_12cmd_original.pt
```
## PhyAug for KWS
A microphone can be characterized by its frequency response. We proposed a fast microphone profiling by playing back and record white noise using microphones, then use white noise to estimate a microphone's frequency response curve (FRC).

The white noise data is downloadable along with microphone data and placed in 
`\white_noise` folder. It contains original white noise data and corresponding microphone white noise. Each white noise data is around 5-minutes long. 

To apply PhyAug for KWS, only original dataset and microphone white noise are needed.
- set `use_tf = True` at line 14 in `train_PhyAug.py` to use PhyAug for model training
- You need to modify the directory location at line 554 in `model.py`, point to the location of your speech dataset: `config["data_folder"] = "your/speech_dataset/location"` 

```bash
python train_PhyAug.py --type train
```
The model training time is around 2 hours for 40 epoches on wour workstation.

We provide our pretrained model in `/model` folder. To use it, simply run: 

```bash
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_phyaug.pt
```
The result corresponds to Figure 7 in our paper.