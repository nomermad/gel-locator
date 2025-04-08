import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.express as px
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression # this could be any ML method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error, 
)

import zipfile
from PIL import Image

# Load the image
image_path = "Gel_19.tif"
image = Image.open(image_path)
if image.mode == 'I;16' or image.mode == 'I;16B':
    image_array = np.array(image)
    image_array = (255 * (image_array / image_array.max())).astype('uint8')
    image = Image.fromarray(image_array, mode='L')



st.title("GEL Classification APP")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section", ["Introduction","Data Overview", "Predictions", "GEL Locater"])

if option == "Introduction":
    st.title("Overview of GEL Electrophoresis")
    st.write("DNA gel electrophoresis is an important technique in biology laboratories that separates DNA fragments by size. However, accurately identifying and analyzing the regions of interest (ROI’s) is difficult, due to the background noise and variations in image quality.A ground truth dataset was established by manually creating binary image masks for a collection of DNA gel images. The ground truth is significant in building the model. It serves as a reference when comparing the performance of various models. This project explores thresholding methods, convolutional neural networks, and random forest. ")
    st.title("Overview of this App")
    st.markdown("""
    To determine which technique accurately reproduces the ground truth masks, three different techniques were used - thresholding and CNN.
    
    Thresholding was utilized to see if the ground truth image could be recreated using a simple statistical method. Niblack thresholding is useful for images with variations, which makes it suitable for image segmentation. Niblack thresholding calculates a threshold for each pixel based on the mean and standard deviation of the surrounding neighborhood. The image is divided into non-overlapping windows, and the mean and standard deviation of the pixels are calculated for each window. A threshold is then determined for each window. Niblack Thresholding classies pixels with thresholds higher than the local as foreground, and pixels with thresholds lower as background. The local mean and standard deviation provides an estimate for the mean level by the amount of local deviation. The formula also includes a k value, where k is a tuning parameter that controls the threshold sensitivity. To optimize the threshold results, a grid search method was used to determine the best k-value. The intersection over union(IoU) loss function was the selection criteria. The IoU loss function finds the overlap between predicted and ground truth area, and minimizes the difference between them. It divides the area of intersection by the area of union, to calculate the difference. The optimal k value will maximize the IoU score, which ensures the best niblack threshold. 

    
   ResU-Net was utilized to see if a neural network can perform better than the threshold results when accurately reproducing the ground truth masks. As mentioned previously, the ResU-Net model uses both U-Net and ResNet structures. ResU-Net uses the encoder-decode structure, which allows the model to skip connections and retain spatial information, improving on the accuracy of the segmentation. Moreover, ResNet’s residual blocks address the vanishing gradient problem.""") 

elif option == "Data Overview":
    st.title("Overview of the Images")
    st.write("""The dataset analyzed was a set of images that were found online and through prior experience in a lab. Initially, there were about 20 images that were then split into smaller segments. After splitting the images, there were 1000 images that were then picked based on their significance. There ended up being 50 training images and 5 testing images. The images represent PCR gel images. A PCR gel image represents the results of a PCR experiment conducted using gel electrophoresis. Gel electrophoresis separates the DNA fragments by size, which allows one to see if and what size DNA was amplified. 
    To clean the dataset, noise was filtered out of the images. The size of the images was also reduced.""")
    st.title("PCR Gel Image Viewer - Raw Image")
    st.image(image, caption="Gel 19", use_column_width=True)

elif option == "Predictions":
    st.write("INSERT HERE")

elif option == "GEL Locater":
    st.write("INSERT HERE")

    
    
    

    



