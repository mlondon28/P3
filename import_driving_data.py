import csv
import cv2
import numpy as np

lines = []
with open('driving_data/combined_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print("--- CSV data loaded. ---")

car_images = []
steering_angles = []
i = 0

def process_image(img):
    # NORMALIZE IMAGE
    # img = (img / 255.0 - 0.5)
    # CROP IMAGE
	# Image cropped in lambda layer
    return img

def augment_data(images, measurements):
    augmented_images, augmented_measurements = [], []
    print("Augmenting dataset...")

    # print("len images, measurements: {}, {}".format(len(images), len(measurements)))
    for image, measurement in zip(images, measurements):
        # try:
        #     print("------Image type: {}".format(type(image)))
        #     print("------Measurement type: {}".format(type(measurement)))
        #     print("images.shape: {}".format(image.shape))
        #     # print("image[0]:  {}".format(len(image[0])))
        # except:
        #     print(image)
        #     break
        # try:
        # print("type(image): {}".format(type(image)))
        # print(image)
        # print(measurement)
        try:
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(measurement * -1.0)
            augmented_images.append(image)
            augmented_measurements.append(measurement)
        except:
            print("image None error occurred.")

        # except:
        #     print("Image could not be flipped!")
        #     print("type(image): {}".format(type(image)))
        # 	# print("augmented measurements len: {}".format(len(augmented_measurements)))
    try:    
        print("Dataset Augmented!")
        #print("type(augmented_images): {}".format(type(augmented_images)))
        #print("type(augmented_images[0]: {}".format(type(augmented_images[0])))
        print("len(augmented_measurements): {}".format(len(augmented_measurements)))
    except:
        pass
    return augmented_measurements

for line in lines:
    # create adjusted steering measurements for the side camera images
    correction = 0.25 # this is a parameter to tune
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'driving_data/IMG/' + filename
        Image = cv2.imread(current_path)

        if i == 0:
            img_center = process_image(np.asarray(Image))
        elif i == 1:
            img_left = process_image(np.asarray(Image))
        else:
            img_right = process_image(np.asarray(Image))    

    # add images and angles to data set
    car_images.extend((img_center, img_left, img_right))
    steering_angles.extend((steering_center, steering_left, steering_right))


# add try|except statement.
# if a line is not a np array, do not add it to the list?
try: 
    print("car_images[0].shape: {}".format(car_images[0].shape))
    print("steering_angles[0].shape: {}".format(steering_angles[0].shape))    
    print("Images extracted and dataset augmented")
    print("augmented_images[0].shape : {}, augmented_measurements[0].shape: {} ".format(augmented_images[0].shape))
    print("augment_images.shape: {}".format(augmented_images.shape))


except:
    print("image has no shape")
augmented_measurements = augment_data(car_images, steering_angles)

X_train = np.load('X_train.npy') 
y_train = np.array(augmented_measurements)

print("Starting Network Training.")
print("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
import gc
gc.collect()
