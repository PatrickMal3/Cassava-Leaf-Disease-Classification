# imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
import cv2
from PIL import Image
import albumentations as alb
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# constants
leaf_df = pd.read_csv('~/Projects/cassava_data/train.csv')
leaf_image_directory = '../cassava_data/train_images'

validate_split = 0.2
batch_size = 8

img_width = 300
img_height = 300

input_shape = (img_height, img_width, 3)

#####
# define class weights
#####

class_nums = []
for x in range(5):
    num = leaf_df[leaf_df['label'] == x].count()[0]
    class_nums.append(num)

class_weights = {}
for x in range(5):
    class_weights[x] = max(class_nums)/class_nums[x]

print(class_weights)

#####
# create train and validate dataframes
#####

# establish amounts for train/validate split
num_images = leaf_df.shape[0]
num_valid = int(num_images*validate_split)
num_train = num_images - num_valid

# shuffle data
shuffled_df = leaf_df.sample(frac=1)

# create validation and traning dataframes
valid_df = shuffled_df.iloc[:num_valid,:]
train_df = shuffled_df.iloc[num_valid+1:,:]

valid_x = valid_df['image_id'].tolist()
valid_x = [leaf_image_directory + "/" + element for element in valid_x]
valid_y = valid_df['label'].tolist()
train_x = train_df['image_id'].tolist()
train_x = [leaf_image_directory + "/" + element for element in train_x]
train_y = train_df['label'].tolist()


#####
# define augmentation function
#####
transform = alb.Compose([
                         alb.RandomCrop(width=300, height=300, p=1.0),
                         alb.HorizontalFlip(),
                         alb.VerticalFlip(),
                         alb.Rotate(),
                         alb.GaussianBlur(p=0.3),
                         alb.CoarseDropout(min_holes=1,
                                           max_holes=20,
                                           min_height=20,
                                           max_height=50,
                                           min_width=20,
                                           max_width=50),
                         alb.RandomBrightnessContrast(),
                         alb.ToGray(p=0.3),
                         #alb.HueSaturationValue(),
                         alb.RandomBrightnessContrast(p=0.3),
                       ])

def augment(img):
    #np_img = np.array(img)
    aug_img = transform(image=img)['image']
    return aug_img

## test the preprocessing function
#img = Image.open('../cassava_data/train_images/7288550.jpg')
#print(img.size)
###img = img.resize((300,300))
#img = np.array(img)
#augmented= augment(img)
#plt.imshow(img)
#plt.show()
#plt.imshow(augmented)
#plt.show()


#####
# define data generator functions
#####

# training values
train_names = train_df['image_id'].values
train_labels = train_df['label'].values

# validation values
valid_names = valid_df['image_id'].values
valid_labels = valid_df['label'].values

# train generator
def train_generator():
    while True:
        for start in range(0, num_train, batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, num_train-1)
            for img_path in range(start, end):
                img = Image.open(train_x[img_path])
                #img = img.resize((img_width, img_height))
                img_array = np.array(img)
                img_array = augment(img_array)
                x_batch.append(img_array)
                y_batch.append(train_y[img_path])

            yield (np.array(x_batch), np.array(y_batch))


# valid generator
def valid_generator():
    while True:
        for start in range(0, num_valid, batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, num_valid-1)
            for img_path in range(start, end):
                img = Image.open(train_x[img_path])
                #img = img.resize((img_width, img_height))
                img_array = np.array(img)
                img_array = augment(img_array)
                x_batch.append(img_array)
                y_batch.append(valid_y[img_path])

            yield (np.array(x_batch), np.array(y_batch))



# define pretrained model
pre_trained_model = EfficientNetB0(
                                   weights=None, 
                                   include_top=False,
                                   input_shape=input_shape,
                                  )

pre_trained_model.load_weights('efficientnetb0_notop.h5')
pre_trained_model.trainable = True

dropout_rate = 0.3

# define model
model = Sequential()
model.add(pre_trained_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(dropout_rate))
model.add(Dense(5, activation="softmax"))

# compile model
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# summarize model
#model.summary()

#model.fit(ds_train, epochs=1, class_weight=class_weights)

# fit the model
model.fit(
    train_generator(),
    epochs= 3,
    steps_per_epoch= num_train // batch_size,
    validation_data= valid_generator(),
    validation_steps = num_valid // batch_size,
    class_weight=class_weights,
)   
