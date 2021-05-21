# -*- coding: utf-8 -*-
"""MapBiomas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AFcTJ4M-Vl5IRxf49EfPUXo9BCY0QBqT
"""


#imports 

import rasterio 
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
import pandas as pd
import numpy as np
from matplotlib import image as im
from matplotlib import pyplot as plt
from affine import Affine
from keras import layers
from keras import Input
import keras
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.metrics import MeanIoU

#MapBiomas label 
label_path = "/content/drive/MyDrive/Honors/MapBiomas/COLECAO_5_DOWNLOADS_COLECOES_ANUAL_AMAZONIA_AMAZONIA-2019.tif"

#a portion of the original satellite image
image_path = "/content/drive/MyDrive/Honors/MapBiomas/MAPBIOMAS-EXPORT/input.tif"


#these are now the coordinates to which we fit the label
minx, miny = -55.80990315107875, -6.617709035051692
maxx, maxy = -54.936201705744104, -7.514587014716622
bbox = box(minx, miny, maxx, maxy)

#insert it into a GeoDataFrame
geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
geo = geo.to_crs(crs=label.crs.data)

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

#get the coordinates
coords = getFeatures(geo)

#clip the raster with the polygon of proper coordinates
label = rasterio.open(label_path)
out_label, out_transform = mask(label, shapes=coords, crop=True)

#copy the metadata that needs to be modified after the clipping
out_meta = label.meta.copy()

#finally update the metadata
out_meta.update({"driver": "GTiff",
                  "height": out_label.shape[1],
                  "width": out_label.shape[2],
                  "transform": out_transform,
                  "crs": pycrs.parse.from_epsg_code(4326).to_proj4()} #the epsg code is the same as from the original meta
                          )

#save the clipped file 
with rasterio.open(label_path, "w", **out_meta) as dest:
  dest.write(out_label)


"""# Data generator"""

def data_gen(src, lab, for_train: bool):
    '''
    generate labels to feed to the U-net
    filter the image using its several bands
    change the value for_train to False to use it for test data
    batch size could be added aswell, this version is used for debug for now
    '''
    
    #list of the selected bands
    bands = [1,2,3,4,5] # for example

    split_thresh = 0.8
    width = 12 # 158 = width/256
    height = 13 # 221 = height/256

    train_width = int(width*split_thresh)
    
    if for_train:
        for train_i in range(train_width):
            for train_j in range(height):
                train_window = rasterio.windows.Window(int(train_i*256), int(train_j*256), 256, 256)
                img_crop = np.zeros((256, 256, len(bands)))
                # filter the bands
                for i, band in enumerate(bands):
                    img_crop[:,:,i] = src.read(band, window=train_window)
                label_crop = lab.read(1, window=train_window).reshape((256, 256, 1))
                
                yield (img_crop, label_crop)
                
    else:
        for test_i in range(train_width, width): # the rest 20% part of the image
            for test_j in range(height):
                test_window = rasterio.windows.Window(int(test_i*256), int(test_j*256), 256, 256)
                img_crop = np.zeros((256, 256, len(bands)))
                for i, band in enumerate(bands):
                    img_crop[:,:,i] = src.read(band, window=test_window)
                label_crop = lab.read(1, window=train_window).reshape((256, 256, 1))
                
                yield (img_crop, label_crop)

def make_batches(size, src, lab, for_train):
  """
  converts the python generator into a tf.data.Dataset object
  and builds batches for of data
  """
  gen = lambda: (row for row in data_gen(src, lab, for_train)) #to make the generator a callable for the line bellow
  dataset = Dataset.from_generator(gen, output_types=(tf.float32, tf.float32))
  dataset = dataset.shuffle(156) #shuffle as many as there are samples
  dataset = dataset.repeat(2)    #repeat some samples at most twice
  dataset = dataset.batch(size)  
  return(dataset)

def build_model(input_layer, start_neurons):
    """
    makes a U-net with 4 down- and up-sampling layers with the functional API
    """
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.5)(pool1)

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

input_layer = Input(shape=(256,256,1))
output_layer = build_model(input_layer, 64)

model = keras.Model(inputs=input_layer, outputs=output_layer)

#the best and most used metric for image segmentation we use Intersection over Union
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 0.001),
    loss='categorical_crossentropy', 
    metrics=MeanIoU(num_classes=42))

label = rasterio.open(label_path)
image = rasterio.open(image_path)

model.fit(make_batches(10, image, label, for_train = True), validation_data = make_batches(10, image, label, for_train=False), epochs=2)

