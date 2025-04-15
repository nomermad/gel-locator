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

from PIL import Image
import numpy as np

images = {}
for i in range(1, 21):
    image_path = f"Gel_{i}.tif"
    image = Image.open(image_path)
    if image.mode == 'I;16' or image.mode == 'I;16B':
        image_array = np.array(image)
        image_array = (255 * (image_array / image_array.max())).astype('uint8')
        image = Image.fromarray(image_array, mode='L')
    images[f'image_{i}'] = image

# Access with images['image_1'], images['image_2'], ..., images['image_20']


image_path_otsu_1 = "Gel_1_otsu.tif"
image_otsu_1 = Image.open(image_path_otsu_1)
if image_otsu_1.mode == 'I;16' or image_otsu_1.mode == 'I;16B':
    image_array_otsu_1 = np.array(image_otsu_1)
    image_array_otsu_1 = (255 * (image_array_otsu_1 / image_array_otsu_1.max())).astype('uint8')
    image_otsu_1 = Image.fromarray(image_array_otsu_1, mode='L')

image_path_otsu_3 = "Gel_3_otsu.tif"
image_otsu_3 = Image.open(image_path_otsu_3)
if image_otsu_3.mode == 'I;16' or image_otsu_3.mode == 'I;16B':
    image_array_otsu_3 = np.array(image_otsu_3)
    image_array_otsu_3 = (255 * (image_array_otsu_3 / image_array_otsu_3.max())).astype('uint8')
    image_otsu_3 = Image.fromarray(image_array_otsu_3, mode='L')

image_basic_1 = "Gel_1.tif"
image_basic_1 = Image.open(image_basic_1)
if image_basic_1.mode == 'I;16' or image_basic_1.mode == 'I;16B':
    image_array_basic_1 = np.array(image_basic_1)
    image_array_basic_1 = (255 * (image_array_basic_1 / image_array_basic_1.max())).astype('uint8')
    image_basic_1 = Image.fromarray(image_array_basic_1, mode='L')

image_basic_3 = "Gel_3.tif"
image_basic_3 = Image.open(image_basic_3)
if image_basic_3.mode == 'I;16' or image_basic_3.mode == 'I;16B':
    image_array_basic_3 = np.array(image_basic_3)
    image_array_basic_3 = (255 * (image_array_basic_3 / image_array_basic_3.max())).astype('uint8')
    image_basic_3 = Image.fromarray(image_array_basic_3, mode='L')

image_mask_1 = "Mask_1.tif"
image_mask_1 = Image.open(image_mask_1)
if image_mask_1.mode == 'I;16' or image_mask_1.mode == 'I;16B':
    image_array_mask_1 = np.array(image_mask_1)
    image_array_mask_1 = (255 * (image_array_mask_1 / image_array_mask_1.max())).astype('uint8')
    image_mask_1 = Image.fromarray(image_array_mask_1, mode='L')

image_mask_3 = "Mask_3.tif"
image_mask_3 = Image.open(image_mask_3)
if image_mask_3.mode == 'I;16' or image_mask_3.mode == 'I;16B':
    image_array_mask_3 = np.array(image_mask_3)
    image_array_mask_3 = (255 * (image_array_mask_3 / image_array_mask_3.max())).astype('uint8')
    image_mask_3 = Image.fromarray(image_array_mask_3, mode='L')

image_basic_1_127 = "Gel_1_127.tif"
image_basic_1_127 = Image.open(image_basic_1_127)
if image_basic_1_127.mode == 'I;16' or image_basic_1_127.mode == 'I;16B':
    image_array_basic_1_127 = np.array(image_basic_1_127)
    image_array_basic_1_127 = (255 * (image_array_basic_1_127 / image_array_basic_1_127.max())).astype('uint8')
    image_basic_1_127 = Image.fromarray(image_array_basic_1_127, mode='L')

image_basic_3_127 = "Gel_3_127.tif"
image_basic_3_127 = Image.open(image_basic_3_127)
if image_basic_3_127.mode == 'I;16' or image_basic_3_127.mode == 'I;16B':
    image_array_basic_3_127 = np.array(image_basic_3_127)
    image_array_basic_3_127 = (255 * (image_array_basic_3_127 / image_array_basic_3_127.max())).astype('uint8')
    image_basic_3_127 = Image.fromarray(image_array_basic_3_127, mode='L')

image_path_phansalker_1 = "Gel_1_phansalker.tif"
image_phansalker_1 = Image.open(image_path_phansalker_1)
if image_phansalker_1.mode == 'I;16' or image_phansalker_1.mode == 'I;16B':
    image_array_phansalker_1 = np.array(image_phansalker_1)
    image_array_phansalker_1 = (255 * (image_array_phansalker_1 / image_array_phansalker_1.max())).astype('uint8')
    image_phansalker_1 = Image.fromarray(image_array_phansalker_1, mode='L')

image_path_phansalker_3 = "Gel_3_phansalker.tif"
image_phansalker_3 = Image.open(image_path_phansalker_3)
if image_phansalker_3.mode == 'I;16' or image_phansalker_3.mode == 'I;16B':
    image_array_phansalker_3 = np.array(image_phansalker_3)
    image_array_phansalker_3 = (255 * (image_array_phansalker_3 / image_array_phansalker_3.max())).astype('uint8')
    image_phansalker_3 = Image.fromarray(image_array_phansalker_3, mode='L')

image_path_niblack_1 = "Gel_1_niblack.tif"
image_niblack_1 = Image.open(image_path_niblack_1)
if image_niblack_1.mode == 'I;16' or image_niblack_1.mode == 'I;16B':
    image_array_niblack_1 = np.array(image_niblack_1)
    image_array_niblack_1 = (255 * (image_array_niblack_1 / image_array_niblack_1.max())).astype('uint8')
    image_niblack_1 = Image.fromarray(image_array_niblack_1, mode='L')

image_path_niblack_3 = "Gel_3_niblack.tif"
image_niblack_3 = Image.open(image_path_niblack_3)
if image_niblack_3.mode == 'I;16' or image_niblack_3.mode == 'I;16B':
    image_array_niblack_3 = np.array(image_niblack_3)
    image_array_niblack_3 = (255 * (image_array_niblack_3 / image_array_niblack_3.max())).astype('uint8')
    image_niblack_3 = Image.fromarray(image_array_niblack_3, mode='L')

image_path_niblack_binary_1 = "binary_Niblack_layer1.tif"
image_niblack_binary_1 = Image.open(image_path_niblack_binary_1)
if image_niblack_binary_1.mode == 'I;16' or image_niblack_binary_1.mode == 'I;16B':
    image_array_niblack_binary_1 = np.array(image_niblack_binary_1)
    image_array_niblack_binary_1 = (255 * (image_array_niblack_binary_1 / image_array_niblack_binary_1.max())).astype('uint8')
    image_niblack_binary_1 = Image.fromarray(image_array_niblack_binary_1, mode='L')

image_path_27 = "Gel_27.tif"
image_27 = Image.open(image_path_27)
if image_27.mode == 'I;16' or image_27.mode == 'I;16B':
    image_array_27 = np.array(image_27)
    image_array_27 = (255 * (image_array_27 / image_array_27.max())).astype('uint8')
    image_27 = Image.fromarray(image_array_27, mode='L')

image_mask_27 = "Mask_27.tif"
image_mask_27 = Image.open(image_mask_27)
if image_mask_27.mode == 'I;16' or image_mask_27.mode == 'I;16B':
    image_array_mask_27 = np.array(image_mask_27)
    image_array_mask_27 = (255 * (image_array_mask_27 / image_array_mask_27.max())).astype('uint8')
    image_mask_27 = Image.fromarray(image_array_mask_27, mode='L')

image_path_reconstructed_27 = "reconstructed_Gel_27.tiff"
image_reconstructed_27 = Image.open(image_path_reconstructed_27)
if image_reconstructed_27.mode == 'I;16' or image_reconstructed_27.mode == 'I;16B':
    image_array_reconstructed_27 = np.array(image_reconstructed_27)
    image_array_reconstructed_27 = (255 * (image_array_reconstructed_27 / image_array_reconstructed_27.max())).astype('uint8')
    image_reconstructed_27 = Image.fromarray(image_array_reconstructed_27, mode='L')

section_path_025 = "section_025.tiff"
section_025 = Image.open(section_path_025)
if section_025.mode == 'I;16' or section_025.mode == 'I;16B':
    section_array_025 = np.array(section_025)
    section_array_025 = (255 * (section_array_025 / section_array_025.max())).astype('uint8')
    section_025 = Image.fromarray(section_array_025, mode='L')

section_path_045 = "section_045.tiff"
section_045 = Image.open(section_path_045)
if section_045.mode == 'I;16' or section_045.mode == 'I;16B':
    section_array_045 = np.array(section_045)
    section_array_045 = (255 * (section_array_045 / section_array_045.max())).astype('uint8')
    section_045 = Image.fromarray(section_array_045, mode='L')

section_path_niblack_025 = "binary_mask_test_niblack.tif"
section_niblack_025 = Image.open(section_path_niblack_025)
if section_niblack_025.mode == 'I;16' or section_niblack_025.mode == 'I;16B':
    section_array_niblack_025 = np.array(section_niblack_025)
    section_array_niblack_025 = (255 * (section_array_niblack_025 / section_array_niblack_025.max())).astype('uint8')
    section_niblack_025 = Image.fromarray(section_array_niblack_025, mode='L')

section_path_niblack_045 = "binary_mask_test2_Niblack.tif"
section_niblack_045 = Image.open(section_path_niblack_045)
if section_niblack_045.mode == 'I;16' or section_niblack_045.mode == 'I;16B':
    section_array_niblack_045 = np.array(section_niblack_045)
    section_array_niblack_045 = (255 * (section_array_niblack_045 / section_array_niblack_045.max())).astype('uint8')
    section_niblack_045 = Image.fromarray(section_array_niblack_045, mode='L')

section_path_test_025 = "binary_mask_test.tif"
section_test_025 = Image.open(section_path_test_025)
if section_test_025.mode == 'I;16' or section_test_025.mode == 'I;16B':
    section_array_test_025 = np.array(section_test_025)
    section_array_test_025 = (255 * (section_array_test_025 / section_array_test_025.max())).astype('uint8')
    section_test_025 = Image.fromarray(section_array_test_025, mode='L')

section_path_test_045 = "binary_mask_test2.tif"
section_test_045 = Image.open(section_path_test_045)
if section_test_045.mode == 'I;16' or section_test_045.mode == 'I;16B':
    section_array_test_045 = np.array(section_test_045)
    section_array_test_045 = (255 * (section_array_test_045 / section_array_test_045.max())).astype('uint8')
    section_test_045 = Image.fromarray(section_array_test_045, mode='L')

image_path_background_gel = "background-gel.tif"
image_background_gel = Image.open(image_path_background_gel)
if image_background_gel.mode == 'I;16' or image_background_gel.mode == 'I;16B':
    image_background_gel_array = np.array(image_background_gel)
    image_background_gel_array = (255 * (image_background_gel_array / image_background_gel_array.max())).astype('uint8')
    image_background_gel = Image.fromarray(image_background_gel_array, mode='L')

image_path_positive_gel = "positive-gel.tif"
image_positive_gel = Image.open(image_path_positive_gel)
if image_positive_gel.mode == 'I;16' or image_positive_gel.mode == 'I;16B':
    image_positive_gel_array = np.array(image_positive_gel)
    image_positive_gel_array = (255 * (image_positive_gel_array / image_positive_gel_array.max())).astype('uint8')
    image_positive_gel = Image.fromarray(image_positive_gel_array, mode='L')

image_path_augmented_gel = "augmented-gel.tif"
image_augmented_gel = Image.open(image_path_augmented_gel)
if image_augmented_gel.mode == 'I;16' or image_augmented_gel.mode == 'I;16B':
    image_augmented_gel_array = np.array(image_augmented_gel)
    image_augmented_gel_array = (255 * (image_augmented_gel_array / image_augmented_gel_array.max())).astype('uint8')
    image_augmented_gel = Image.fromarray(image_augmented_gel_array, mode='L')

image_path_iou_results = "iou-results.tif"
image_iou_results = Image.open(image_path_iou_results)
if image_iou_results.mode == 'I;16' or image_iou_results.mode == 'I;16B':
    image_iou_results_array = np.array(image_iou_results)
    image_iou_results_array = (255 * (image_iou_results_array / image_iou_results_array.max())).astype('uint8')
    image_iou_results = Image.fromarray(image_iou_results_array, mode='L')

image_path_randomforest_results = "randomforest-results.tif"
image_randomforest_results = Image.open(image_path_randomforest_results)
if image_randomforest_results.mode == 'I;16' or image_randomforest_results.mode == 'I;16B':
    image_randomforest_results_array = np.array(image_randomforest_results)
    image_randomforest_results_array = (255 * (image_randomforest_results_array / image_randomforest_results_array.max())).astype('uint8')
    image_randomforest_results = Image.fromarray(image_randomforest_results_array, mode='L')

image_path_xgboost_results = "xgboost-results.tif"
image_xgboost_results = Image.open(image_path_xgboost_results)
if image_xgboost_results.mode == 'I;16' or image_xgboost_results.mode == 'I;16B':
    image_xgboost_results_array = np.array(image_xgboost_results)
    image_xgboost_results_array = (255 * (image_xgboost_results_array / image_xgboost_results_array.max())).astype('uint8')
    image_xgboost_results = Image.fromarray(image_xgboost_results_array, mode='L')

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section", ["Introduction","Data Overview", "Methods","Thresholding & Res-U-Net", "GEL Locator"])

if option == "Introduction":
    st.title("Overview of GEL Electrophoresis")
    st.write("DNA gel electrophoresis is an important technique in biology laboratories that separates DNA fragments by size. However, accurately identifying and analyzing the regions of interest (ROI’s) is difficult, due to the background noise and variations in image quality.A ground truth dataset was established by manually creating binary image masks for a collection of DNA gel images. The ground truth is significant in building the model. It serves as a reference when comparing the performance of various models. This project explores thresholding methods, convolutional neural networks, and random forest. ")
    st.title("Overview of this App")
    st.markdown("""
    To determine which technique accurately reproduces the ground truth masks, three different techniques were used - thresholding and CNN.""") 

elif option == "Data Overview":
    st.title("Overview of the Images")
    st.write("""The dataset analyzed was a set of images that were found online and through prior experience in a lab. Initially, there were about 20 images that were then split into smaller segments. After splitting the images, there were 1000 images that were then picked based on their significance. There ended up being 50 training images and 5 testing images. The images represent PCR gel images. A PCR gel image represents the results of a PCR experiment conducted using gel electrophoresis. Gel electrophoresis separates the DNA fragments by size, which allows one to see if and what size DNA was amplified. 
    To clean the dataset, noise was filtered out of the images. The size of the images was also reduced.
    """)
    image_selection = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1",
                                                 "Gel 2",
                                                 "GEL 3",
                                                 "GEL 4",
                                                 "Gel 4",
                                                 "GEL 6",
                                                "GEL 7",
                                                 "Gel 8",
                                                 "GEL 9",
                                                 "GEL 10",
                                                 "Gel 11",
                                                 "GEL 12",
                                                 "GEL 13",
                                                 "Gel 14",
                                                 "GEL 15",
                                                 "GEL 16",
                                                 "Gel 17",
                                                 "GEL 18",
                                                "Gel 19",
                                                 "GEL 20"])
    if image_selection == "GEL 1":
        st.title("PCR Gel Image Viewer - Gel 1")
        st.image(images['image_1'], caption="Gel 1", use_column_width=True)

    elif image_selection == "GEL 2":
        st.title("PCR Gel Image Viewer - Gel 2")
        st.image(images['image_2'], caption="Gel 2", use_column_width=True)

    elif image_selection == "GEL 3":
        st.title("PCR Gel Image Viewer - Gel 3")
        st.image(images['image_3'], caption="Gel 3", use_column_width=True)

    elif image_selection == "GEL 4":
        st.title("PCR Gel Image Viewer - Gel 4")
        st.image(images['image_4'], caption="Gel 4", use_column_width=True)

    elif image_selection == "GEL 5":
        st.title("PCR Gel Image Viewer - Gel 5")
        st.image(images['image_5'], caption="Gel 5", use_column_width=True)

    elif image_selection == "GEL 6":
        st.title("PCR Gel Image Viewer - Gel 6")
        st.image(images['image_6'], caption="Gel 6", use_column_width=True)

    elif image_selection == "GEL 7":
        st.title("PCR Gel Image Viewer - Gel 7")
        st.image(images['image_7'], caption="Gel 7", use_column_width=True)

    elif image_selection == "GEL 8":
        st.title("PCR Gel Image Viewer - Gel 8")
        st.image(images['image_8'], caption="Gel 8", use_column_width=True)

    elif image_selection == "GEL 9":
        st.title("PCR Gel Image Viewer - Gel 9")
        st.image(images['image_9'], caption="Gel 9", use_column_width=True)

    elif image_selection == "GEL 10":
        st.title("PCR Gel Image Viewer - Gel 10")
        st.image(images['image_10'], caption="Gel 10", use_column_width=True)

    elif image_selection == "GEL 11":
        st.title("PCR Gel Image Viewer - Gel 11")
        st.image(images['image_11'], caption="Gel 11", use_column_width=True)

    elif image_selection == "GEL 12":
        st.title("PCR Gel Image Viewer - Gel 12")
        st.image(images['image_12'], caption="Gel 12", use_column_width=True)

    elif image_selection == "GEL 13":
        st.title("PCR Gel Image Viewer - Gel 13")
        st.image(images['image_13'], caption="Gel 13", use_column_width=True)

    elif image_selection == "GEL 14":
        st.title("PCR Gel Image Viewer - Gel 14")
        st.image(images['image_14'], caption="Gel 14", use_column_width=True)

    elif image_selection == "GEL 15":
        st.title("PCR Gel Image Viewer - Gel 15")
        st.image(images['image_15'], caption="Gel 15", use_column_width=True)

    elif image_selection == "GEL 16":
        st.title("PCR Gel Image Viewer - Gel 16")
        st.image(images['image_16'], caption="Gel 16", use_column_width=True)

    elif image_selection == "GEL 17":
        st.title("PCR Gel Image Viewer - Gel 17")
        st.image(images['image_17'], caption="Gel 17", use_column_width=True)

    elif image_selection == "GEL 18":
        st.title("PCR Gel Image Viewer - Gel 18")
        st.image(images['image_18'], caption="Gel 18", use_column_width=True)

    elif image_selection == "GEL 19":
        st.title("PCR Gel Image Viewer - Gel 19")
        st.image(images['image_19'], caption="Gel 19", use_column_width=True)

    elif image_selection == "GEL 20":
        st.title("PCR Gel Image Viewer - Gel 20")
        st.image(images['image_20'], caption="Gel 20", use_column_width=True)

elif option == "Methods":
    st.title("Overview of Methodology")
    st.write("""Thresholding was utilized to see if the ground truth image could be recreated using a simple statistical method. Image thresholding separates an image into two or more regions based on the pixel intensity value. Image thresholding simplifies a grayscale image into a binary image based on the image intensity level compared to the threshold value. This reduces the image to two levels of intensity. Global and local thresholding methods were used. 

Global thresholding is a segmentation technique that uses a single threshold value to categorize all pixels in an image into foreground or background. The native method was a base threshold model set at 125. A base threshold splits all pixels in the images as a 1 or 0 based on its histogram intensity value. Otsu’s method is a global thresholding technique used for image segmentation. It separates an image into two classes, foreground and background, based on the pixel's grayscale intensity values. Otsu’s method detects an optimal threshold value that separates the two regions using the grayscale histogram. The two regions separate into maximum variance. 

Moreover, local thresholding uses unique threshold values for the partitioned subimages obtained from the whole image.Local thresholding calculates a threshold value for each region based on local features, such as the mean intensity. Mean adaptive thresholding calculates the threshold value for each sub image by taking the average intensity of all the pixels in that area. A constant value was then subtracted from this mean to get the final threshold value. Phansalker local thresholding adapts the threshold value for each pixel based on its local neighbor. Niblack thresholding is useful for images with variations, which makes it suitable for image segmentation. Niblack thresholding calculates a threshold for each pixel based on the mean and standard deviation of the surrounding neighborhood. The image is divided into non-overlapping windows, and the mean and standard deviation of the pixels are calculated for each window. A threshold is then determined for each window. Niblack Thresholding classies pixels with thresholds higher than the local as foreground, and pixels with thresholds lower as background. The local mean and standard deviation provides an estimate for the mean level by the amount of local deviation. The formula also includes a k value, where k is a tuning parameter that controls the threshold sensitivity. To optimize the niblack threshold results, a grid search method was used to determine the best k-value. The intersection over union(IoU) loss function was the selection criteria. The IoU loss function finds the overlap between predicted and ground truth area, and minimizes the difference between them. It divides the area of intersection by the area of union, to calculate the difference. The optimal k value will maximize the IoU score, which ensures the best niblack threshold. 

CNN is a type of deep learning algorithm that is good for analyzing visual data by using a technique called convolution to extract features. In this project, ResU-Net(Residual U-Net) is used. ResU-Net combines the U-Net architecture with residual networks(resnets) and is specifically designed for image segmentation. U-Net is a CNN architecture designed for image segmentation, and is noted by its encoder-decoder structure that skips connections to allow for increase of detailed information from the encoder to the decoder. ResNet is a CNN architecture as well, but focuses on solving the vanishing gradient problem by using skip connections. By combining U-Net and ResNet, ResU-Net attempts to strengthen both architectures.  

A Gel locator was created to help locate each segment of the output images. The gel locator works by analyzing the segmented binary masks to determine the exact position and boundaries of each gel band. The gel locator initially identifies potential gel regions. The detected regions are then filtered based on their size and shape characteristics and the regions are classified as either valid gel segments or noise. The coordinates of each identified gel segment are extracted and bounding boxes are then generated. The gel locator was evaluated on the output of the models, using the test dataset. 
""")

elif option == "Thresholding & Res-U-Net":
    st.title("Thresholding")
    st.write("**Masked images for Thresholding**")
    mask_image_selection_basic = st.selectbox("Choose an Image to display:", 
                                                ["Mask 1",
                                                 "Mask 3"])
    if mask_image_selection_basic == "Mask 1":
        st.title("PCR Gel Image Viewer - Mask 1")
        st.image(image_mask_1, caption="Mask 1", use_column_width=True)

    elif mask_image_selection_basic == "Mask 3":
        st.title("PCR Gel Image Viewer - Mask 3")
        st.image(image_mask_3, caption="Mask 3", use_column_width=True)
    st.write("**Original Images before Thresholding**")
    original_image_selection_basic = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1",
                                                 "GEL 3"])
    if original_image_selection_basic == "GEL 1":
        st.title("PCR Gel Image Viewer - Gel 1")
        st.image(image_basic_1, caption="Gel 1", use_column_width=True)

    elif original_image_selection_basic == "GEL 3":
        st.title("PCR Gel Image Viewer - Gel 3")
        st.image(image_basic_3, caption="Gel 3", use_column_width=True)

    st.write("**Images after Otsu Thresholding**")
    otsu_image_selection_basic = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1 Otsu",
                                                 "GEL 3 Otsu"])
    if otsu_image_selection_basic == "GEL 1 Otsu":
        st.title("PCR Gel Image Viewer - Gel 1 Otsu")
        st.image(image_otsu_1, caption="Gel 1 Otsu", use_column_width=True)

    elif otsu_image_selection_basic == "GEL 3 Otsu":
        st.title("PCR Gel Image Viewer - Gel 3 Otsu")
        st.image(image_otsu_3, caption="Gel 3 Otsu", use_column_width=True)

    st.write("**Images after Basic Thresholding**")
    image_selection_basic_127 = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1 127",
                                                 "GEL 3 127"])
    if image_selection_basic_127 == "GEL 1 127":
        st.title("PCR Gel Image Viewer - Gel 1 127")
        st.image(image_basic_1_127, caption="Gel 1 127", use_column_width=True)

    elif image_selection_basic_127 == "GEL 3 127":
        st.title("PCR Gel Image Viewer - Gel 3 127")
        st.image(image_basic_3_127, caption="Gel 3 127", use_column_width=True)

    st.write("**Images after Phansalker Thresholding**")
    phansalker_image_selection_basic = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1 Phansalker",
                                                 "GEL 3 Phansalker"])
    if phansalker_image_selection_basic == "GEL 1 Phansalker":
        st.title("PCR Gel Image Viewer - Gel 1 Phansalker")
        st.image(image_phansalker_1, caption="Gel 1 Phansalker", use_column_width=True)

    elif phansalker_image_selection_basic == "GEL 3 Phansalker":
        st.title("PCR Gel Image Viewer - Gel 3 Phansalker")
        st.image(image_phansalker_3, caption="Gel 3 Phansalker", use_column_width=True)
    st.write("**Images after Niblack Thresholding**")
    niblack_image_selection_basic = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1 Niblack",
                                                 "GEL 3 Niblack"])
    if niblack_image_selection_basic == "GEL 1 Niblack":
        st.title("PCR Gel Image Viewer - Gel 1 Niblack")
        st.image(image_niblack_1, caption="Gel 1 Niblack", use_column_width=True)

    elif niblack_image_selection_basic == "GEL 3 Niblack":
        st.title("PCR Gel Image Viewer - Gel 3 Niblack")
        st.image(image_niblack_3, caption="Gel 3 Niblack", use_column_width=True)

    st.title("**Res-U-Net**")
    st.write("**Fragmented sections**")
    fragment_selection_original = st.selectbox("Choose an Image to display:", 
                                                ["Selection 025",
                                                 "Selection 045"])
    if fragment_selection_original == "Selection 025":
        st.title("PCR Gel Image Viewer - Selection 025")
        st.image(section_025, caption="Selection 025", use_column_width=True)

    elif fragment_selection_original == "Selection 045":
        st.title("PCR Gel Image Viewer - Selection 045")
        st.image(section_045, caption="Selection 045", use_column_width=True)

    st.write("**Fragmented sections - Test**")
    fragment_selection_test = st.selectbox("Choose an Image to display:", 
                                                ["Selection 025 - Test",
                                                 "Selection 045 - Test"])
    if fragment_selection_test == "Selection 025 - Test":
        st.title("PCR Gel Image Viewer - Selection 025 - Test")
        st.image(section_test_025, caption="Selection 025 - Test", use_column_width=True)

    elif fragment_selection_test == "Selection 045 - Test":
        st.title("PCR Gel Image Viewer - Selection 045 - Test")
        st.image(section_test_045, caption="Selection 045 - Test", use_column_width=True)

    st.write("**Fragmented sections - Niblack**")
    fragment_selection_niblack = st.selectbox("Choose an Image to display:", 
                                                ["Selection 025 - Niblack",
                                                 "Selection 045 - Niblack"])
    if fragment_selection_niblack == "Selection 025 - Niblack":
        st.title("PCR Gel Image Viewer - Selection 025 - Niblack")
        st.image(section_niblack_025, caption="Selection 025 - Niblack", use_column_width=True)

    elif fragment_selection_niblack == "Selection 045 - Niblack":
        st.title("PCR Gel Image Viewer - Selection 045 - Niblack")
        st.image(section_niblack_045, caption="Selection 045 - Niblack", use_column_width=True)
   
    
    st.write("**Reconstructed full images**")
    reconstruction_image_selection = st.selectbox("Choose an Image to display:", 
                                                ["GEL 27",
                                                 "GEL 27 Masked",
                                                "Reconstructed Gel 27"])
    if reconstruction_image_selection == "GEL 27":
        st.title("PCR Gel Image Viewer - GEL 27")
        st.image(image_27, caption="Gel 27", use_column_width=True)

    elif reconstruction_image_selection == "GEL 27 Masked":
        st.title("PCR Gel Image Viewer - Gel 27 Masked")
        st.image(image_mask_27, caption="GEL 27 Masked", use_column_width=True)
    elif reconstruction_image_selection == "Reconstructed Gel 27":
        st.title("PCR Gel Image Viewer - Reconstructed Gel 27")
        st.image(image_reconstructed_27, caption="GEL 27 Reconstructed", use_column_width=True)

    
elif option == "GEL Locator":
    st.title("GEL Images after Band Detection")

    gel_band_selection = st.selectbox("Choose an Image to display:", 
                                                ["Positive Samples",
                                                 "Background Samples",
                                                "Augmented Samples"])
    if gel_band_selection == "Positive Samples":
        st.title("Positive Samples")
        st.image(image_positive_gel, caption="Positive Samples", use_column_width=True)

    elif gel_band_selection == "Background Samples":
        st.title("Background Samples")
        st.image(image_background_gel, caption="Background Samples", use_column_width=True)

    elif gel_band_selection == "Augmented Samples":
        st.title("Augmented Samples")
        st.image(image_augmented_gel, caption="Augumented Samples", use_column_width=True)

    st.title("Random Forest and XGboost on Band Detection")
    st.title("Augmented Samples")
    st.image(image_iou_results, caption="IOU Results", use_column_width=True)

    model_selection = st.selectbox("Choose an Image to display:", 
                                                ["Random Forest",
                                                 "XGboost"])
    if model_selection == "Random Forest":
        st.title("Random Forest")
        st.image(image_randomforest_results, caption="Random Forest", use_column_width=True)

    elif model_selection == "XGboost":
        st.title("XGboost")
        st.image(image_xgboost_results, caption="XGboost", use_column_width=True)
    

    
    

    
    
    
    




    
    
    

    




