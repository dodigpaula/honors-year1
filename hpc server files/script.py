import rasterio
import pandas as pd
import numpy as np
from affine import Affine
import keras
from keras import layers
from keras import Input
import tensorflow as tf
from tensorflow.data import Dataset

label_path = "/mnt/server-home/TUE/20190876/data/output.tif"
image_path = "/mnt/server-home/TUE/20190876/data/input.tif"

kwargs = {
    'crs': {'init': 'epsg:4326'},
    'affine': Affine(0.00026949458523585647, 0.0, -55.80990315107875,
       0.0, -0.00026949458523585647, -6.617709035051692),
    'count': 1,
    'dtype': rasterio.int32,
    'driver': 'GTiff',
    'width': 3242,
    'height': 3328,
    'nodata': None
}


def data_gen(src, lab, for_train: bool):
    '''
    generate labels to feed to the U-net
    filter the image using its several bands
    change the value for_train to False to use it for test data
    batch size could be added aswell, this version is used for debug for now
    '''
    
    #list of the selected bands
    bands = [2,4,6,14,35] # for example

    split_thresh = 0.8
    width = 158 # 158 = width/512
    height = 221 # 221 = height/512

    train_width = int(width*split_thresh)
    
    if for_train:
        for train_i in range(train_width):
            for train_j in range(height):
                train_window = rasterio.windows.Window(int(train_i*512),int(train_j*512), int((train_i*512)+512), int((train_j*512)+512))
                img_crop = np.zeros((512, 512, len(bands)))
                # filter the bands
                for i, band in enumerate(bands):
                    img_crop[:,:,i] = src.read(band, window=train_window)
                label_crop = lab.read(1, window=train_window).reshape((512, 512, 1))
                
                yield (img_crop, label_crop)
                
    else:
        for test_i in range(train_width, width): # the rest 20% part of the image
            for test_j in range(height):
                test_window = rasterio.windows.Window(int(test_i*512),int(test_j*512), int((test_i*512)+512), int((test_j*512)+512))
                img_crop = np.zeros((512, 512, len(bands)))
                for i, band in enumerate(bands):
                    img_crop[:,:,i] = src.read(band, window=test_window)
                label_crop = lab.read(1, window=train_window).reshape((512, 512, 1))
                
                yield (img_crop, label_crop)


def build_model(input_layer, start_neurons):
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    deconv4 = layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

input_layer = Input(shape=(512,512,1))
output_layer = build_model(input_layer, 64)


############## compile the model
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 0.001),
    loss='categorical_crossentropy'
    )

############## load the data
out = rasterio.open(label_path, **kwargs)
image = rasterio.open(image_path)

############## run the model
model.fit(data_gen(image, out, for_train = True), validation_data = data_gen(image, out, for_train=False), epochs=2)

