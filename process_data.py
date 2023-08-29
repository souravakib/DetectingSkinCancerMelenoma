import pandas as pd
import os
import shutil
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import autokeras as ak
import cv2


# Dump all images into a folder and specify the path:
#data_dir = os.getcwd() + "/data/all_images"
data_dir = os.path.abspath('all_images')
#print(data_dir)
#files = glob.glob(os.path.join(data_dir,"**","*"),recursive=True)
#files=[f for f in files if os.path.isfile(f)]
#print(files)
# Path to destination directory where we want subfolders
dest_dir = os.getcwd() + "/data/reorganized/"

# Read the csv file containing image names and corresponding labels
skin_df2 = pd.read_csv('C:/E_Drive/kaBHOOM/UPM/FYP/codes/HAM10000_metadata.csv')


label=skin_df2['dx'].unique().tolist()  #Extract labels into a lisT
label_images = []


# Copy images to new folders
for i in label:
    file_name = dest_dir + str(i) + "/"
    os.makedirs(file_name, exist_ok =True)
    sample = skin_df2[skin_df2['dx'] == i]['image_id']
    label_images.extend(sample)
    print(label_images)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg"))
    label_images = []
    
    

