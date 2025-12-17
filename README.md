# DE-3DGAN: 3D GAN for EEG-based Emotion Recognition
A PyTorch implementation of a 3D Generative Adversarial Network framework for emotion recognition from EEG signals.The algorithm is derived from the article ["DE-3DGAN: A Differential Entropy-Driven 3D Generative Framework for EEG-Based Emotion Recognition"](https://cmsfiles.s3.amazonaws.com/ner2025/proceedings/61569_CFP25CNE-ART/pdfs/0000381.pdf)
## Overview
DE-3DGAN is an innovative 3D GAN framework designed for EEG-based emotion recognition. The system consists of three progressive training stages that enhance classifier performance through high-quality synthetic EEG generation.
## Features
**Three-Stage Training Pipeline**: Sequential GAN training, classifier pre-training, and semi-supervised fine-tuning
**3D U-Net Generator**: Generates realistic EEG signals from differential entropy (DE) features
**Multi-Scale Classifier**: Incorporates attention mechanisms for enhanced feature extraction
**Differential Entropy Features**: Utilizes DE features as prior knowledge for signal generation
**Improved Generalization**: Synthetic data augmentation boosts classifier robustness
## Data Preparation
The experiment is based on the public dataset DEAP, The data used has been downsampled to 128Hz.
Place data in: ./eeg_dataset/DEAP/data_preprocessed_python/
The DE features after dataset processing are saved in './DE_feature/'
Preprocessing Pipeline: Raw EEG signal to time-frequency representation; Baseline removal; Time-frequency to spatial-frequency conversion; Channel position mapping(8 * 9)
## Quick Start(binary classification task based on valence dimension)
### Stage 1: Train GAN
Train the 3D GAN to generate EEG signals from DE features:
```Bash
python 3D-UNet_GAN_Valence.py
```
### Stage 2: Pre-train Classifier
Pre-train the emotion classifier on real EEG data:
```Bash
python SA-ST_Net_Classifier_Valence.py --fold 0
```
(fold range 0-4)
### Stage 3: Fine-tune with Generated Data
Fine-tune classifier using both real and synthetic data:
```Bash
python Finetune_Classifier_Valence.py --fold 0
```
(fold range 0-4)
## Other Tasks
In addition to the valence dimension, the project includes the implementation of other task types, including binary classification tasks based on arousal labels and four classification tasks based on a mixture of valence and arousal labels, as shown in the table of contents /American main 'and' /Four_Class-main'
