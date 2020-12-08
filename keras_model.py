import pandas as pd
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt
import os 
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, GlobalAveragePooling2D
from PIL import Image
import tensorflow_hub as hub
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# load info and define categories
leaf_info = pd.read_csv('~/Projects/cassava_data/train.csv')
leaf_image_directory = '../cassava_data/train_images'

leaf_disease_categories = ['0', '1', '2', '3', '4']

# determine class weights
class_nums = []
for x in range(5):
    num = leaf_info[leaf_info['label'] == x].count()[0]
    class_nums.append(num)

class_weights = {}
for x in range(5):
    class_weights[x] = max(class_nums)/class_nums[x]

print('Class Weights:')
print(class_weights)
leaf_info['label'] = leaf_info['label'].astype(str)
leaf_info['image_id'] = leaf_info['image_id'].astype(str)



# define preprocessing/augmentation function
transform = alb.Compose([
                         #alb.RandomCrop(width=300, height=300, p=1.0),
                         alb.HorizontalFlip(),
                         alb.VerticalFlip(),
                         alb.Rotate(),
                         alb.GaussianBlur(),
                         #alb.CoarseDropout(min_holes=1,
                         #                  max_holes=20,
                         #                  min_height=20,
                         #                  max_height=50,
                         #                  min_width=20,
                         #                  max_width=50),
                         alb.RandomBrightnessContrast(),
                         #alb.ToGray(),
                         #alb.HueSaturationValue()
                       ])

def augment(img):
    np_img = np.array(img)
    aug_img = transform(image=np_img)['image']
    return aug_img

# test the preprocessing function
#img = Image.open('/storage/mydata/cassava_data/train_images/0/292918886.jpg')
#print(img.size)
#img = img.resize((300,300))
#augmented= augment(img)
#plt.imshow(img)
#plt.show()
#plt.imshow(augmented)
#plt.show()

batch_size = 8
train_input_shape = (300,300,3)
n_classes = len(leaf_disease_categories)

train_datagen = ImageDataGenerator(
                                   preprocessing_function=augment,
                                   validation_split=0.2,
                                   rescale=1./255.
                                  )

train_generator = train_datagen.flow_from_dataframe(
                                                    directory=leaf_image_directory,
                                                    dataframe=leaf_info,
                                                    x_col='image_id',
                                                    y_col='label',
                                                    target_size=train_input_shape[0:2],
                                                    color_mode="rgb",
                                                    class_mode='categorical',
                                                    classes=leaf_disease_categories,
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                   )

valid_generator = train_datagen.flow_from_dataframe(
                                                    directory=leaf_image_directory,
                                                    dataframe=leaf_info,
                                                    x_col='image_id',
                                                    y_col='label',
                                                    target_size=train_input_shape[0:2],
                                                    color_mode="rgb",
                                                    class_mode='categorical',
                                                    classes=leaf_disease_categories,
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                   )

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),
             EarlyStopping(monitor='val_loss', patience=4),
             ModelCheckpoint(filepath='best_model.h5', 
                             monitor='val_loss', 
                             save_best_only=True)]


pre_trained_model = EfficientNetB0(
                                   weights=None, 
                                   include_top=False, 
                                   input_shape=train_input_shape
                                  )

pre_trained_model.load_weights("efficientnetb0_notop.h5")
pre_trained_model.trainable = True

dropout_dense_layer = 0.3


model = Sequential()
model.add(pre_trained_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(5, activation="softmax"))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#model.summary()

print(class_weights)
print(class_weights)
print(class_weights)
print(class_weights)
print(class_weights)
print(class_weights)
print(class_weights)
model.fit(
          train_generator, 
          epochs = 20, 
          validation_data = valid_generator,
          class_weight = class_weights,
          callbacks=callbacks
         ) 
