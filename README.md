# Voice-Activity-Detection

## Introduction

This repo contains two labs in the course of Intelligent speech interaction.

## Requirements

* [Python3](https://www.python.org/)
* [Librosa](http://librosa.github.io/librosa/)
* [Numpy](http://www.numpy.org/)
* [sklearn](http://scikit-learn.org/)

**Windows**

	pip install librosa numpy sklearn

## Contents

### Lab1

Materials: wav file en_4092_a.wav, en_4092_b.wav

Aim: This project uses a simple speech endpoint detection algorithm to process wav files,aiming at deciding if a segment is silent or not.

usage:

	python final.py
	python transfer.py

### Lab2

This lab uses a Machine learning approach(GMM) to achieve the same purpose as the lab1

If you want to use your own wav file,please install HCopy first!

You need to extract features with HCopy using the config file config.feat,which extracts **MFCC** features


	HCopy -C config.feat -S feats.scp

Then you can verify the features exist(*.mfcc),here l have already generated the mfcc file of the given wav files.

usage:

	python filter.py
	python GMM.py