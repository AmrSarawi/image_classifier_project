# Image Classifier Project

This repository contains an image classification project developed as part of Udacity’s *Intro to Machine Learning with TensorFlow Nanodegree Program*. The project demonstrates how to build, train, and deploy a deep learning model for classifying images.

## Overview

The objective of this project is to develop a neural network capable of classifying images of flowers into one of 102 categories using the Oxford 102 Flower Dataset. The trained model is then integrated into a command-line application to perform predictions on new images.

This project represents a complete machine learning workflow, from data preprocessing to model deployment.

## Dataset

The model is trained on the Oxford 102 Flower Dataset, which includes:
- 102 distinct flower categories  
- Approximately 20 images per category  
- Training, validation, and testing splits  

> Note: The dataset is not included in this repository due to its size and must be downloaded separately.

## Methodology

The project follows a structured pipeline:

1. Data loading and preprocessing (resizing and normalization)  
2. Model construction using TensorFlow/Keras  
3. Training and validation of the model  
4. Model evaluation and performance analysis  
5. Saving the trained model  
6. Deployment via a command-line interface  

## Features

- End-to-end image classification pipeline  
- TensorFlow-based neural network implementation  
- Top-K prediction support  
- Model saving and loading  
- Command-line interface for inference  

## Usage

Once the model has been trained and saved, predictions can be generated from the command line using the provided inference script. The script accepts an input image and a trained model file as arguments. Additional optional parameters include `--top_k`, which specifies the number of most likely predictions to return (default is 5), and `--category_names`, which maps class indices to human-readable labels. Example usage: `python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3 --category_names label_map.json`.
