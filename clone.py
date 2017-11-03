import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = lines[1:] #get rid of header

correction = 0.7 #tune

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    current_path = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(current_path) #this reads bgr, drive may send RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (160, 80)) #size down
                    angle = float(batch_sample[3])
                    images.append(image)
                    if i==1:
                        angle = angle + correction #left
                    elif i==2:
                        angle = angle - correction #right
                    measurements.append(angle)

                    image_flipped = cv2.flip(image, 1)
                    angle_flipped = angle * -1.0
                    images.append(image_flipped)
                    measurements.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,160,3)))
model.add(Cropping2D(cropping=((25,10), (0,0)), input_shape=(3,80,160)))

#nvidia 
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.7)) #tune
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = \
                len(train_samples), validation_data=validation_generator, \
                nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
exit()

#todo: try dropout, verify the cropping of the image, make more test data
