################################
 # Importing required libraries
################################

import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image 
from typing import List
from keras.metrics import MeanIoU
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.metrics import MeanIoU
from keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

####################################
 # Defining all function to be used
####################################

# Function to load dataset
def load_dataset(root_directory):
    
    # Patch size of each image
    patch_size = 256

    # Array to store images from the directory
    image_dataset = []  

    # Reading and storing images
    for path, subdirs, files in os.walk(root_directory):
        subdirs.sort()
        dirname = path.split(os.path.sep)[-1]
        
        if dirname == 'images':           
            images = os.listdir(path)              
            images.sort()
            
            # Reading and patching images
            for i, image_name in enumerate(images):  
                
                if image_name.endswith(".jpg"):         
                    image = cv2.imread(path+"/"+image_name, 1)            
                    
                    # All images are of different sizes : crop them to nearest integer and divide all images into patches of 256*256*3
                    SIZE_X = (image.shape[1]//patch_size)*patch_size 
                    SIZE_Y = (image.shape[0]//patch_size)*patch_size 
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))
                    image = np.array(image)         
        
                    # Extract patches from each image
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
            
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            
                            single_patch_img = patches_img[i,j,:,:] 
                            single_patch_img = single_patch_img[0]                               
                            image_dataset.append(single_patch_img)
                    

    # Array to store masks from the directory
    mask_dataset = []  

    # Reading and storing masks
    for path, subdirs, files in os.walk(root_directory):
        
        subdirs.sort()
        dirname = path.split(os.path.sep)[-1]
        
        if dirname == 'masks':   
            masks = os.listdir(path)  
            masks.sort()
            
            # Reading and patching masks
            for i, mask_name in enumerate(masks):  
                
                if mask_name.endswith(".png"):   
                    mask = cv2.imread(path+"/"+mask_name, 1) 
                    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                    
                    # All masks are of different sizes : crop them to nearest integer and divide all masks into patches of 256*256*3
                    SIZE_X = (mask.shape[1]//patch_size)*patch_size 
                    SIZE_Y = (mask.shape[0]//patch_size)*patch_size 
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y)) 
                    mask = np.array(mask)

                    # Extract patches from each image
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            
                            single_patch_mask = patches_mask[i,j,:,:]
                            single_patch_mask = single_patch_mask[0]                         
                            mask_dataset.append(single_patch_mask) 

    return image_dataset, mask_dataset

# Function for image processing
def find_edges(images):   
    for i, image in enumerate(images):
        """
        Edge detection takes place in the following setps:
        1. Convert RGB image to greyscale
        2. Apply Gaussian blur to remove noise and unwanted information
        3. Apply Canny Edge Detector for edge detection 
        4. Thresholding (OTSU thresholding used here) and Dialation
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (13, 13), 0)
        canny = cv2.Canny(blur,50, 100)
        ret, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((2,2),np.uint8)
        dilated = cv2.dilate(thresh,kernel,iterations = 1)
        img = np.zeros_like(image)
        img[:,:,0] = dilated
        final = cv2.add(img,image)
        images[i] = final

# Converting Dataset to Greyscale Images
def greyscaling(dataset):
    gray_dataset = []

    for i in dataset:
        # Load the edge-extracted image
        img = i

        # Check if the image is grayscale
        if len(img.shape) == 2:
            # Save the grayscale image directly
            gray_dataset.append(img)
            #cv2.imwrite('grayscale_image.png', img)
        else:
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Save the grayscale image
            gray_dataset.append(gray_img)
            #cv2.imwrite('grayscale_image.png', gray_img)
    return gray_dataset

# Function for K-Means Clustering
def k_means_clustering(data: np.ndarray, k: int, max_iterations: int = 1000) -> np.ndarray:

    # Initialize centroids randomly.
    centroids = data[np.random.choice(data.shape[0], size=k, replace=False), :]

    for iteration in range(max_iterations):
        # Assign each data point to the nearest centroid.
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignment = np.argmin(distances, axis=0)

        # Update the centroid of each cluster to be the mean of all data points assigned to it.
        for i in range(k):
            centroids[i] = np.mean(data[cluster_assignment == i], axis=0)

    return cluster_assignment

# Function for K-Means segmentation using Texture Features
def segment_img(images1):
    patch_groups = []
    for z in images1:
        prop_all = []
        prop_f= np.zeros((1024,24))
        for i in range(32):
            for j in range(32):
                im = z[i:i+8, j:j+8]
                glcm = graycomatrix(im, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                prop_f[i*32 + j] = [graycoprops(glcm, 'contrast')[0][0],graycoprops(glcm, 'contrast')[0][1],graycoprops(glcm, 'contrast')[0][2],graycoprops(glcm, 'contrast')[0][3],graycoprops(glcm, 'dissimilarity')[0][0],graycoprops(glcm, 'dissimilarity')[0][1],graycoprops(glcm, 'dissimilarity')[0][2],graycoprops(glcm, 'dissimilarity')[0][3],graycoprops(glcm, 'homogeneity')[0][0],graycoprops(glcm, 'homogeneity')[0][1],graycoprops(glcm, 'homogeneity')[0][2],graycoprops(glcm, 'homogeneity')[0][3],graycoprops(glcm, 'energy')[0][0],graycoprops(glcm, 'energy')[0][1],graycoprops(glcm, 'energy')[0][2],graycoprops(glcm, 'energy')[0][3],graycoprops(glcm, 'correlation')[0][0],graycoprops(glcm, 'correlation')[0][1],graycoprops(glcm, 'correlation')[0][2],graycoprops(glcm, 'correlation')[0][3],graycoprops(glcm, 'ASM')[0][0],graycoprops(glcm, 'ASM')[0][1],graycoprops(glcm, 'ASM')[0][2],graycoprops(glcm, 'ASM')[0][3]]
        patch_groups.append(k_means_clustering(prop_f, 6, 1000))
    return patch_groups

# Function for Data Augmentation
def data_augmentation(train_data_x, train_data_y):
    length = len(train_data_x)
    for i in range(length):
        seed = np.random.randint(low=0, high=1000)
        datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)
        input_image = datagen.random_transform(train_data_x[i],seed = seed)
        output_image = datagen.random_transform(train_data_y[i],seed = seed)
        train_data_x.append(input_image)
        train_data_y.append(output_image)

# Scale Pixel values of each image using MinMaxScaler
scaler = MinMaxScaler()
def scale_image_pixels(images):
    for i,single_patch_img in enumerate(images):
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        images[i] = single_patch_img

# Function to convert HEX value of categories into integer labels in masks
def label_masks(mask_dataset):

    #Convert the HEX value to RGB value
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) 

    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) 

    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) 

    Vegetation = '#FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) 

    Water = '#E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) 

    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) 

    # Function to convert RGB value of categories into integer labels
    def rgb_to_2D_label(label):

        label_seg = np.zeros(label.shape,dtype=np.uint8)
        label_seg [np.all(label==Building,axis=-1)] = 0
        label_seg [np.all(label==Land,axis=-1)] = 1
        label_seg [np.all(label==Road,axis=-1)] = 2
        label_seg [np.all(label==Vegetation,axis=-1)] = 3
        label_seg [np.all(label==Water,axis=-1)] = 4
        label_seg [np.all(label==Unlabeled,axis=-1)] = 5
        
        label_seg = label_seg[:,:,0] 
        
        return label_seg

    # Array to store labelled masks
    labels = []

    # Converting RGB to integer labels in masks
    for i in range(len(mask_dataset)):
        label = rgb_to_2D_label(mask_dataset[i])
        labels.append(label)    

    # Converting labels to numpy array
    labels = np.array(labels)  
    labels = np.expand_dims(labels, axis=3)

    # Return number of classed and categorical labels
    n_classes = len(np.unique(labels))
    labels_cat = to_categorical(labels, num_classes=n_classes)

    return n_classes, labels_cat

# Define accuracy metric
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# Create the U-Net Semantic Segmentation Network
def multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2) 
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def get_model():
    return multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)

##########################
 # Execution and training
##########################

# Root Directory of dataset
root_directory = '/home/priyanshjain/Downloads/GNR602/Semantic segmentation dataset'

# Load dataset
image_dataset, mask_dataset = load_dataset(root_directory)

# Split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 42)

# Find Edges in the training images
find_edges(X_train)

# Converting Image Dataset to greyscale images
gray_img_dataset=greyscaling(X_train)

# Finding Texture Features and Segmentation using K-means Clustering
K_Means_Segmented = segment_img(gray_img_dataset)

# Data Augmentation to obtain a larger dataset
data_augmentation(X_train, y_train)

# Scale each pixel value between 0 and 1
scale_image_pixels(X_train)

# Convert masks to labels and images to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
n_classes, y_train = label_masks(y_train)
n_classes, y_test = label_masks(y_test)

# Train the Semantic Segmentation Network
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss) 
metrics=['accuracy', jacard_coef]

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train, 
                    batch_size = 20, 
                    verbose=1, 
                    epochs=30,
                    validation_data=(X_test, y_test), 
                    shuffle=False)
model.save('models/U-net.hdf5')

#######
 # END
#######


