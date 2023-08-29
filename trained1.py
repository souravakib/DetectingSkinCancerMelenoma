import os
import pandas as pd
import shutil
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import autokeras as ak
from sklearn.model_selection import train_test_split
import cv2

# Read the csv file containing image names and corresponding labels
skin_df2 = pd.read_csv('C:/E_Drive/kaBHOOM/UPM/FYP/codes/HAM10000_metadata.csv')

# Modify the 'dx' column to have binary labels: 1 for 'mel', 0 for other classes
skin_df2['dx'] = skin_df2['dx'].apply(lambda x: 1 if x == 'mel' else 0)

# Split the dataframe into training and validation subsets
train_df, val_df = train_test_split(skin_df2, test_size=0.2, stratify=skin_df2['dx'], random_state=42)

# Set up the paths and load the data using ImageDataGenerator
data_dir = os.path.abspath('all_images')
img_height, img_width = 299, 299
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="image_id",
    y_col="dx",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw',
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=data_dir,
    x_col="image_id",
    y_col="dx",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw',
)

# Train the model using AutoKeras
input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    block_type="efficient",
    augment=True,
    normalize=True,
    )(input_node)
output_node = ak.ClassificationHead(num_classes=1, activation="sigmoid")(output_node)

clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=5,
)

clf.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
)

# Save and load the model
model = clf.export_model()
model.save("melanoma_detection_model.h5")
model = load_model("melanoma_detection_model.h5", custom_objects=ak.CUSTOM_OBJECTS)

# Set up the webcam integration and test the model
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (img_height, img_width))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0

    prediction = model.predict(frame)
    predicted_class = int(prediction[0] > 0.5)

    if predicted_class == 1:
        label = 'melanoma'
    else:
        label = 'non-melanoma'

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Melanoma Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
