# PhyAug for Seismic localization
This is seismic localization case study in IPSN'21 paper: [PhyAug: Physics-Directed Data Augmentationfor Deep Sensing Model Transfer inCyber-Physical Systems](https://arxiv.org/pdf/2104.01160.pdf).

This repo consists of `/matlab_code_gen_data` folder for data generation and `/python_localization` folder for fingerprinting based seismic event localization. `/matlab_code_gen_data` code is based on [locally-sparse-tomography](https://github.com/mikebianco/locally-sparse-tomography). **The credit goes to the corresponding authors**.

## Implementation details
We apply PhyAug for seismic localization based on scientific settings. Specifically, time difference of arrival (TDoA) of a seismic event is used as input features, we adopt fingerprinting based approach for localization. The model implmentation details can be found in our paper.

## Dataset
Dataset use in our experiment is generated in `/matlab_code_gen_data`. To generate same dataset, open the folder with Matlab, then execute file `conventional_run_phyaug_gen_invert_data.m`. This file need to be executed twice:  set `use_invert = true` in line 53 to generate true data using `true slowness model` and set `use_invert = false` in line 53 to generate augmented data using `estimated slowness model`.
It takes ~8hours to generate whole set of data on our workstation. We provide a set of data used in our experiment and is downloadable in this [link](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KFMI1K). 
Dataset need to put in `python_localization/data` folder.
```

## Installation
- To run Matlab code, Matlab software need to install on workstation, we use Matlab 2019b in this experiment
- To run python code, install the python packages use `pip`
```bash
pip install -r seismic_requirement.txt
```
## Train/Eval model
We implement supporting vector machine (SVM) and multi-layer perceptron (MLP) for classification. The code is placed in `python_localization` folder, make sure you are in this folder when training model.

### Experiment result on 2% perturbation
In this setting, 2% of perturbation noise is added to sensor arrival time.
Train model on true data, set `model = 'svm'` at line 40 to train SVM, set `model = 'MLP'` to train MLP:
```bash
python evaluate_2pct_noise_shmoo_true_data.py
```
Train model on augmented data, set `model = 'MLP'` at line 40 to train SVM, set `model = 'MLP'` to train MLP:
```bash
python evaluate_2pct_noise_shmoo_aug_data.py
```
The result is saved in `.csv` file for processing. The result corresponds to figure 14 in paper.

### Experiment result on various perturbation
We evaluate the model performance on dataset generated using different noise perturbation (1% - 10%)
Train model on true data, set `model = 'svm'` at line 40 to train SVM, set `model = 'MLP'` to train MLP:
```bash
python evaluate_perturb_true_data.py
```
Train model on augmented data, set `model = 'MLP'` at line 40 to train SVM, set `model = 'MLP'` to train MLP:
```bash
python evaluate_perturb_aug_data.py
```
The result is saved in `.csv` file for processing. The result corresponds to figure 16 in our paper.
