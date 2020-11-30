import pandas as pd
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, GlobalAveragePooling2D
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

leaf_info = pd.read_csv('../cassava_data/train.csv')
leaf_image_directory = '../cassava_data/train_images/'
leaf_disease_categories = ['0','1','2','3','4']

# put pictures in corrosponding folders
#leaf_ids = leaf_info['image_id']
#leaf_labels = leaf_info['label']
#leaf_len = len(leaf_ids)
#for x in range(leaf_len):
#    shutil.move(os.path.join(leaf_image_directory, leaf_ids[x]), os.path.join(leaf_image_directory, leaf_labels[x]))


class_nums = []
for x in range(5):
    cur_dir = leaf_image_directory + str(x)
    num_files = len([f for f in os.listdir(cur_dir)if os.path.isfile(os.path.join(cur_dir, f))])
    class_nums.append(num_files)

class_weights = {}
for x in range(5):
    class_weights[x] = max(class_nums)/class_nums[x]

print(class_weights)

#batch_size = 8
#train_input_shape = (300,300,3)
#n_classes = len(leaf_disease_categories)
#
#train_datagen = ImageDataGenerator(validation_split=0.3,
#                                   rescale=1./255.,
#                                  )
#
#train_generator = train_datagen.flow_from_directory(
#                                                    directory=leaf_image_directory,
#                                                    target_size=train_input_shape[0:2],
#                                                    color_mode="rgb",
#                                                    class_mode='categorical',
#                                                    classes=leaf_disease_categories,
#                                                    batch_size=batch_size,
#                                                    subset="training",
#                                                    shuffle=True,
#                                                   )
#
#valid_generator = train_datagen.flow_from_directory(
#                                                    directory=leaf_image_directory,
#                                                    target_size=train_input_shape[0:2],
#                                                    color_mode="rgb",
#                                                    class_mode='categorical',
#                                                    classes=leaf_disease_categories,
#                                                    batch_size=batch_size,
#                                                    subset="validation",
#                                                    shuffle=True,
#                                                   )
#
##model = Sequential()
##model.add(Conv2D(64,3,3,input_shape=(224,224,3)))
##model.add(Activation('relu'))
##model.add(MaxPooling2D(pool_size = (2,2)))
##
##model.add(Conv2D(64,3,3))
##model.add(Activation('relu'))
##model.add(MaxPooling2D(pool_size = (2,2)))
##
##model.add(Flatten())
##model.add(Dense(64))
##
##model.add(Dense(5))
##model.add(Activation('sigmoid'))
##
##model.compile(
##              loss = 'binary_crossentropy',
##              optimizer = 'adam',
##              metrics = ['accuracy']
##             )
#
##mobile net
##URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
##hub_model = hub.KerasLayer(URL, input_shape = (224, 224, 3))
##hub_model.trainable = True
##
##
##model = Sequential()
##model.add(hub_model)
##
##model.add(Dense(5, activation = 'softmax'))
##
##
##
##
#
#effnet_layers = EfficientNetB3(weights=None, include_top=False, input_shape=(300,300,3))
#
#for layer in effnet_layers.layers:
#    layer.trainable = True    
#
#dropout_dense_layer = 0.3
#
#model = Sequential()
#model.add(effnet_layers)
#    
#model.add(GlobalAveragePooling2D())
#model.add(Dense(256))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(dropout_dense_layer))
#
#model.add(Dense(5, activation="softmax"))
#
#model.compile(loss = 'categorical_crossentropy',
#              optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
#              metrics = ['accuracy'])
#
#model.fit(
#          train_generator, 
#          epochs = 10, 
#          validation_data = valid_generator,
#          class_weight = class_weights
#          )
#
