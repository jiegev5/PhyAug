# PhyAug for Acoustic-Based Room Recognition (ARR)

This is PyTorch implementation of ARR case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentation for Deep Sensing Model Transfer in Cyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

## Background
This case study uses smartphones to record rooms' acoustic background spectrogram (ABS), then trains a DNN for room-level recognition.
Our experiment shows that the room recognition accuracy drop can be up to 80% if the pre-trained model is evaluated using the data collected from a specific smartphone microphone. PhyAug recovers the accuracy loss by 33%-80%

## Installation
Set up the virtual environment and use pip to install packages. Noted that we evaluate under python3.6.9:

```bash
pip install -r requirement.txt
```

## dataset
- We record three smartphones (Galaxy s7, Moto Z and Pixel 4) to record 20 rooms' ABS. The training data is 10-minutes long, testing data is 2-minute long. The raw data is in `.pcm` format.
- The original data set is downloadable from this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/M1HDNT).
- We provide the scripts to convert the `.pcm` files to the ABS feature and stored in `.pickle` file.
- We suggest to create a `/data` folder and place the raw data and converted feature inside:
```
/data
    /pcm_data
        # place downloaded pcm files here
    /spec
        # place the converted features here for model training/evaluation
```

## Feature Extraction and Model Evaluation

First, we extract the ABS features from pcm files. Please change the `source_phone` argument in the script and run it for each phone. 
```bash
python 00_extract_ABS.py
```

Next, we train a DNN model using training data from "Galaxy s7", and evaluate the trained model on all smartphones.

```bash
python 10_main_MLP.py
```
It's expected to see ~98\% accuracy for "Galaxy s7" and 16\% - 18\% accuracy on "Pixel 4" and "Moto Z". You can change the `source_phone` in the script to train model on different smartphone's training data.

## PhyAug for ARR
We use smartphones to record a 1-minute ABS in a room to obtain the smartphone's ABS profile. Then use the ABS profiles to transfer the data from a source smartphone to different target smartphones. 

The script for feature extraction and data augmentation use ABS profile is:
```bash
python 01_extract_ABS_PhyAug.py
```
You need to set the `source_phone` and `target_phone` in the script.

Then we can use the augmented data to re-train a DNN and re-evaluate on smartphones' data:

```bash
python 10_main_MLP.py
```
You need to set the `source_phone` and `target_phone` and set `phyAug = True` in the script.

## Calibration and CDA approach for ARR
We also provide the codes of baselines in this case study:

### Calibration

The script for feature extraction:
```bash
python 02_extract_ABS_calibrate.py
```

Train and evaluate model:
```bash
python 11_main_MLP_Calibrate.py
```

### CDA

The script for feature extraction:
```bash
python 03_extract_ABS_CDA.py
```

Train and evaluate model:
```bash
python 12_main_MLP_CDA.py
```

[**Back to front page**](../README.md)