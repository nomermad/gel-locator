# 🧬 Gel Bands: DNA Gel Electrophoresis Analysis
**Team Members: Jack Ruhala, Linn Wang, Maddy Nomer**

## 📋 Overview

Gel Bands is here to change how researchers analyze gel electrophoresis results. This application uses machine learning techniques to detect and analyze DNA fragments in gel images. 

Our approach leverages different techniques such as thresholding, Res-U-Net, and Random Forest. These approaches are designed to handle the challenges of different image quality, background noise, or distortions. Gel Bands will help with classifying the foreground ROIs from the background noise. 


## 🔍 Dataset Development
* **Original dataset**: 20 images from a lab 
* **Diverse sample collection**: Gels with varying quality, resolution, and experimental conditions
* **Augmentation techniques**: The images were split into smaller images and then picked on sigificance

## 🧠 Multi-Model Approach
In this project, we've developed:

1. **Thresholding algorithms** Niblack thresholding optimized for gel images
2. **Convolutional neural networks** trained to identify bands despite background noise
3. **Ensemble methods** (XGBOOST and Random Forest) for classification refinement

By combining these multiple approaches, we aim to create a tool that significantly reduces analysis time while improving reproducibility in molecular biology research.
