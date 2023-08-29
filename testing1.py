# -*- coding: utf-8 -*-
"""
Created on Mon May 15 00:18:09 2023

@author: ssour
"""

import numpy as np
import pandas as pd
import os
import cv2
import json
from instagram_scraper import InstagramScraper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Define image preprocessing function
def preprocess_image(image_path, img_height, img_width):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    return img

# Set image dimensions
img_height, img_width = 32, 32

# Load dataset and metadata
data_dir = 'C:/E_Drive/kaBHOOM/UPM/FYP/codes/all_images'
metadata_csv = 'C:/E_Drive/kaBHOOM/UPM/FYP/codes/HAM10000_metadata.csv'
metadata = pd.read_csv(metadata_csv)

# Preprocess all images in the dataset
images = []
for img_path in metadata['image_path']:
    images.append(preprocess_image(os.path.join(data_dir, img_path), img_height, img_width))
images = np.array(images)

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(metadata['dx'])
labels = to_categorical(labels, num_classes=len(le.classes_))

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# Define the model architecture
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top = False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(le.classes_), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32
history = model.fit(
    X_train, y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(X_val, y_val)
)

# Save the trained model
model.save('skin_cancer_detector.h5')

# Define Instagram scraper
search_hashtag = 'skin_lesion'  # Replace with the desired hashtag
max_images = 100  # Limit the number of images to download
output_dir = 'instagram_images'
scraper = InstagramScraper()
scraper.usernames = [search_hashtag]
scraper.maximum = max_images
scraper.media_metadata = True
scraper.latest = True
scraper.destination = output_dir

# Scrape images from Instagram
scraper.scrape_hashtag()

# Load Instagram metadata
metadata_file = os.path.join(output_dir, f'{search_hashtag}.json')
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Process the scraped images
instagram_images = []
for entry in metadata['GraphImages']:
    image_path = os.path.join(output_dir, entry['filename'])
    instagram_images.append(preprocess_image(image_path, img_height, img_width))
instagram_images = np.array(instagram_images)

# Make predictions on the Instagram images
predictions = model.predict(instagram_images)
predicted_classes = le.inverse_transform(np.argmax(predictions, axis=1))

for i, prediction in enumerate(predicted_classes):
    print(f'Image {i+1}: Predicted diagnosis: {prediction}')
