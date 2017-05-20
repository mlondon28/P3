
# coding: utf-8

# In[39]:

import csv
import cv2
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[47]:

lines = []
with open('driving_data/combined_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print("--- CSV data loaded. ---")

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("Train Samples: {} \nValidation Samples: {}".format(len(train_samples), len(validation_samples)))


# In[41]:

def preprocess_image(img):
    # NORMALIZE IMAGE
    img = img / 255 - 0.5
    # CROP IMAGE
    crop = img[50:140,:,:]
    # Resize to 64x64 ?
    # image_array = cv2.resize(image_array, (64, 64))
    return crop


# In[42]:

# def augment_data(images, measurements):
#     augmented_images, augmented_measurements = [], []
#     print("Augmenting dataset...")
    
#     for image, measurement in zip(images, measurements):
#         try:
#             augmented_images.append(cv2.flip(image,1))
#             augmented_measurements.append(measurement * -1.0)
#             augmented_images.append(image)
#             augmented_measurements.append(measurement)
#         except:
#             print("image None error occurred.")
            
#     try:    
#         print("Dataset Augmented!")
#         print("type(augmented_images): {}".format(type(augmented_images)))
#         print("type(augmented_images[0]: {}".format(type(augmented_images[0])))
#     except:
#         pass
#     return augmented_images, augmented_measurements


# In[58]:

def brighten(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    rand = 0.25 + np.random.uniform()
    img[:,:,2] = img[:,:,2]*rand
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img


# In[44]:

def flip(img, angle):
    img = (cv2.flip(img,1))
    angle = angle * -1.0
    return img, angle


# In[64]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            flipped_img = []
            flipped_angle = []
            steering_angles = []
            for batch_sample in batch_samples:
                correction = 0.23 # this is a parameter to tune
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                steering_angles.extend((center_angle, left_angle, right_angle)) 
#                 print(steering_angles)
                
                for i in range(3):
                    name = 'driving_data/IMG/'+batch_sample[i].split('/')[-1]
                    Image = cv2.imread(name)
                    Image = preprocess_image(Image) # crop and normalize
                    flipped_img, flipped_angle = flip(Image, steering_angles[i])
#                     Image = brighten(Image)  # Randomly brighten image
                    
                    # convert to gray?
                    # possibly leave out some near zero steering angle data
                    if i == 0:
                        img_center = np.asarray(Image)
                        center_flip = np.asarray(flipped_img)
                        c_flip_angle = flipped_angle
                    elif i == 1:
                        img_left = np.asarray(Image)
                        left_flip = np.asarray(flipped_img)
                        l_flip_angle = flipped_angle
                    else:
                        img_right = np.asarray(Image)  
                        right_flip = np.asarray(flipped_img)
                        r_flip_angle = flipped_angle

                try: 
                    noneTest = cv2.flip(img_center,1)
                    noneTest = cv2.flip(img_left,1)
                    noneTest = cv2.flip(img_right,1)
                    images.extend((img_center, img_left, img_right, center_flip, left_flip, right_flip))
                    angles.extend((steering_center, steering_left, steering_right, c_flip_angle, l_flip_angle, r_flip_angle))
                except:
                    print("Exception importing data in generator fcn")
                steering_angles = []
#                 print(len(images))
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[65]:

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 90, 320  # Trimmed image format


# In[69]:

zx, zy = next(train_generator)


# In[71]:

print(len(zx), len(zy))
print(zx.shape)


# In[74]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,input_shape=(90,320,3), activation="relu"), )
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch=192, validation_data=validation_generator, nb_val_samples=192, nb_epoch=3)

model.save('model.h5')
import gc
gc.collect()


# In[ ]:



