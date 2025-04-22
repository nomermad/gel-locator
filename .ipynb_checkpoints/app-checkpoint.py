import streamlit as st
import pandas as pd
import numpy as np
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


st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #000000, #1a1a2e, #16213e, #0f3460);
        color: white;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* General text */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #e0e0e0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        color: #e0e0e0 !important;
    }

    /* Dropdown input box */
    div[data-baseweb="select"] > div {
        background-color: #1c1c1c !important;
        color: #f0f0f0 !important;
        border-color: #444 !important;
    }

    /* Dropdown menu */
    div[data-baseweb="select"] div[role="listbox"] {
        background-color: #1c1c1c !important;
        color: #f0f0f0 !important;
    }

    /* Dropdown options */
    div[data-baseweb="select"] div[role="option"] {
        background-color: #1c1c1c !important;
        color: #f0f0f0 !important;
    }

    /* Hovered dropdown option */
    div[data-baseweb="select"] div[role="option"]:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)




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
    st.write("Image data is a very common type of data collected for biological experiments. This project focuses on the basics of biological image processing focusing analysis of DNA gel electrophoresis images. This project evaluates multiple approaches for segmenting regions of interest (ROIs) containing fluorescent DNA, from the background. The methods include traditional thresholding techniques such as Otsu thresholding as well as advanced approaches like convolutional neural networks (CNNs). Performance evaluation was done using accuracy, jaccard, and recall metrics.")
    st.markdown("""
    A ground truth dataset was established by manually creating binary image masks for a collection of DNA gel images. The ground truth is significant in building the model. It serves as a reference when comparing the performance of various models. This project explores thresholding methods, convolutional neural networks, and random forest. The ultimate goal of this project is to determine which technique accurately reproduces the ground truth masks, which provides researchers with a reliable tool..""") 


elif option == "Data Overview":
    st.title("Overview of the Images")
    st.write("""The dataset analyzed was a set of images that were found online and prior lab data. Each image was converted to 8-bit data for consistency. Each image was manually classified by hand and the labels were transferred to a binary ground truth mask
    """)
    image_selection = st.selectbox("Choose an Image to display:", 
                                                ["GEL 1",
                                                 "GEL 2",
                                                 "GEL 3",
                                                 "GEL 4",
                                                 "GEL 4",
                                                 "GEL 6",
                                                "GEL 7",
                                                 "GEL 8",
                                                 "GEL 9",
                                                 "GEL 10",
                                                 "GEL 11",
                                                 "GEL 12",
                                                 "GEL 13",
                                                 "Gel 14",
                                                 "GEL 15",
                                                 "GEL 16",
                                                 "GEL 17",
                                                 "GEL 18",
                                                "GEL 19",
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
    
    st.write("**Preprocessing for Neural Networks**")
    st.write("Specifically for a neural network to run, the data has to be of a constant size and shape. One common image size used is 256x256. Each image used was split into smaller segments of 256x256 if the original image was larger than the required shape. After splitting the images, there were over 1000 images. Most of the sections generated contained only background, so to balance the data set, 50 images out of the 1000 were then picked based on present significant features.")

    st.write("**Augmentation**")
    st.write("Augmentation was only used on the training data set used for neural network training. The computer would pick a random rotation of between 0 and 45 degrees based on a uniform distribution, and rotate both the image and the corresponding mask. Both images were then saved and used for training. Roughly 50 images from the training data were chosen at random with replacement, to expand the training data set to be around 100 images")

    st.write("**Naive Model**")
    st.write("Our baseline naive model was to threshold each image at the absolute median pixel value of 126.")

    st.write("**Global Thresholding**")
    st.write("Global thresholding is a segmentation technique that applies the same threshold value to all pixels in the image. A global threshold can be set manually or be calculated based on the images overall properties. Otsu is a common method used for global thresholding. Otsu’s method defined the threshold that maximises intra-class variance. Of the foreground and background class.")

    st.write("**Local Thresholding**")
    st.write("""Local thresholding applies a unique threshold value for each sample or pixel, based on the values of neighboring pixels. Local thresholding is commonly used when there exists high class dependents from neighboring points. Local threshold requires an input window to determine how many pixels away from the pixel of interest should impact the threshold of said pixel. One local method is mean thresholding. The mean threshold calculates the threshold value for each pixel by taking the average intensity of all the pixels in an area and assigning class based on if the pixel is higher or lower than the mean. 

More sophisticated local thresholds like Niblack thresholding calculate a threshold based on mean and standard deviation of the neighborhood. Niblack thresholding calculates a threshold for each pixel based on the mean and standard deviation of the surrounding neighborhood. 

Niblack is calculated as t= mN + (k*sdN) where mN and sdN are the mean and  standard deviation of the neighborhood, and k is a constant normally set as k = -0.2. Niblack has a number of variations that give more weight to specific aspects of the neighborhood, one Niblack variation is Phansalker. The Phansalker threshold formula ist= mN *(1+p*exp(-q*mN)+k)*( (sdN/R)-1). The extra parameters included in the Phansalker formula reduce the likelihood of FP getting introduced into the final image.""")
    

    

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
    

    
    

    
    
    
    




    
    
    

    




