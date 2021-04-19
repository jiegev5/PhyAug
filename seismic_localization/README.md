# PhyAug for Seismic localization
This is seismic localization case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentation for Deep Sensing Model Transfer in Cyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

This repo consists of `/matlab_code_gen_data` folder for data generation and `/python_localization` folder for fingerprinting based seismic event localization. 

Note that `/matlab_code_gen_data` code is based on [locally-sparse-tomography](https://github.com/mikebianco/locally-sparse-tomography). **The credit goes to the corresponding authors**.

## Model implementation
We apply PhyAug for seismic localization. Specifically, we adopt fingerprinting based approach for localization. Time difference of arrival (TDoA) of a seismic event is used as input features. We implement supporting vector machine (MLP) and multi-layer perceptron (MLP) for localization. The model implementation details can be found in our paper.

## Dataset
Dataset used in our experiment is generated in `/matlab_code_gen_data`. To obtain the same dataset as used in our experiment, open the folder with Matlab, then execute `conventional_run_phyaug_gen_invert_data.m`. This file needs to be executed twice:  
1. set `use_invert = true` in line 53 to generate data using `true slowness model`, 
2. set `use_invert = false` to generate augmented data using `estimated slowness model`.

It takes around 8 hours to generate a full set of data on our workstation. We provide a set of data used in our experiment and is downloadable from this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KFMI1K). 
Please place them in `python_localization/data` folder, the folder hierarchy looks like this:
```
/seismic_localization
    /matlab_code_gen_data
    /python_localization
        /data
            /inverted_data_sd_1pct_normalized_noise
            /inverted_data_sd_2pct_normalized_noise
                                .
                                .
                                .
            
```

## Installation
- To run Matlab code, Matlab software need to be installed on workstation, we use Matlab 2019b in this experiment
- To run python code, install the python packages using `pip`. Noted that we evaluate our codes under python3.6.9:
```bash
pip install -r seismic_requirement.txt
```
## Train/evaluate model
We implement supporting vector machine (SVM) and multi-layer perceptron (MLP) for classification. The code is placed in `python_localization` folder, make sure you have downloaded the data and placed them in `python_localization/data` before training.

### Experiment result on 2% perturbation
Under this setting, 2% of perturbation noise is added to sensor arrival time.

Train model on data generated from `true slowness model`: set `model = 'svm'` at line 40 to use SVM, set `model = 'MLP'` to use MLP:
```bash
python evaluate_2pct_noise_shmoo_true_data.py
```
Train model on augmented data from `estimated slowness model`: set `model = 'MLP'` at line 90 to use SVM, set `model = 'MLP'` to use MLP:
```bash
python evaluate_2pct_noise_shmoo_aug_data.py
```
The results are saved in `.csv` file for processing. The results correspond to figure 14 in our paper.

### Experiment result on different perturbation noise
We evaluate the model performance on dataset generated using different noise perturbation (1% - 10%).

Train model on data generated from `true slowness model`: set `model = 'svm'` at line 40 to use SVM, set `model = 'MLP'` to use MLP:
```bash
python evaluate_perturb_true_data.py
```
Train model on augmented data from `estimated slowness model`: set `model = 'MLP'` at line 62 to use SVM, set `model = 'MLP'` to use MLP:
```bash
python evaluate_perturb_aug_data.py
```
The results are saved in `.csv` file for processing. The results correspond to figure 16 in our paper.

<center>[**Back to front page**](../README.md)</center>