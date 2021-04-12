# PhyAug for Keyword Spotting

This is PyTorch implementation of keyword spotting (KWS) case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentationfor Deep Sensing Model Transfer inCyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

This Repo is based on [Honk](https://github.com/castorini/honk): A PyTorch reimplementation of Google's TensorFlow convolutional neural networks for keyword spotting. 
**The credit goes to the corresponding authors**.

## Implementation details
This case study applies Convolutional neural network for keyword spotting. Specifically, we apply "cnn-trad-pool2" model to discriminate 12 commands: "yes,no,up,down,left,right,on,off,stop,go,silence and unknown".

## dataset
- The original data set is downloadable from this [link](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html).
- We use five microphones (M1,M2,M3,M4 and M5) to record the white noise and test dataset in a quiet meeting room. The dataset is available from this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/1IM0MD)
- To train/evaluate the model, download the above datasets and put in `/data` folder and organize them following below hierarchy:
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
Set up the virtual environment and use pip to install packages:

```bash
pip install -r PhyAug_kws_requirement.txt
```
## Train model
Train/evaluate script in place in `/utils` folder, please `cd` to this folder and follow below instruction for model training/evaluation.

To train on original dataset:
- set `use_tf = False` and replace `your_model_name.pt` in `train_PhyAug.py`
-  replace `config["data_folder"] = "../data/original_dataset"` in `model.py`

```bash
python train_PhyAug.py --type train
```
## Evaluate model
The model accuracy on original test dataset is 90%. To evaluate model on collected microphone dataset, replace `config["data_folder"] = "../data/recorded_dataset/M1"` in `model.py`. You will see accuracy drop on microphones' dataset. We provide our pretrained model `model_12cmd_original.pt` in `/model` folder.

```bash
python evaluate_model.py --type eval --input_file ../model/model_12cmd_original.pt
```
## PhyAug for KWS
We proposed a fast microphone profiling by playing back white noise and record use microphones. The white noise data is downloadable along with microphone data and placed in 
`\white_noise` folder. It contains original white noise data and corresponding microphone white noise. Each white noise data is ~5mins length. 

To apply PhyAug for KWS, only original dataset and microphone white noise are needed.
- set `use_tf = True` and replace `your_model_name.pt` in `train_PhyAug.py`
- Microphone transfer functions are otained from `tf = mod.get_tf_phyaug(my_t)` in `train_PhyAug.py`, they are used to augment original data during training

```bash
python train_PhyAug.py --type train
```

To evaluate the model, replace the model name and test data folder in `model.py`. We provide our pretrained model in `/model` folder. To use it, simply run: 

```bash
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_phyaug.pt
```

## Hardware specification
The above models are pretrained on a workstation with following specifications:
- CPU: Intel Core i9-7900X 3.30GHz 13.75MB Cache 10C/20T
- GPU: 4x Zotac nVidia RTX2080Ti 11GB GDDR6 PCIe x16 GPU Card