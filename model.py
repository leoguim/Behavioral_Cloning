# Import libraries
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Functions to extract and augment images
def load_records(path):
    records = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            records.append(row)
    return records


def extract_images(images_path, driving_records):
    images = []
    angles = []
    for row in driving_records:
        source_path = row[0]
        filename = source_path.split('/')[-1]   
        current_path = images_path + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)      
        angle = float(row[3])
        angles.append(angle)   
    return images, angles


def augment_images_by_flipping(images, angles):
    augmented_images, augmented_angles = [],[]
    for image, angle in zip(images,angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1)
    return augmented_images, augmented_angles

# Load and Process Images
# 5 sessions have been recorded: Normal center-driving, CounterCW, Reinforcement Areas, Recovery and Curves

folder_path = ['Training_Data', 'Training_Data_CounterCW', 'Training_Data_Recovery',
               'Training_Data_Curves', 'Training_Data_Reinforcement']
images, angles = [], []

for i in range(len(folder_path)):

    records_path = './'+folder_path[i]+'/driving_log.csv'
    images_path = './'+folder_path[i]+'/IMG/'
    print('====> Loading Records for '+folder_path[i] + ' Session')
    driving_records = load_records(records_path)
    print('Total Number of records for '+folder_path[i] + ' Session is: {}'.format(len(driving_records)))
    print('====>  Extracting Images')
    images_, angles_ = extract_images(images_path, driving_records)
    print('Total Number of images for '+folder_path[i] + ' Session is: {}'.format(len(images_)))
    print('#######################################################')
    images.extend(images_)
    angles.extend(angles_)

print('Total Number of images is: {}'.format(len(images)))
print('#######################################################')
print('====>  Augmenting Images')
augmented_images, augmented_angles = augment_images_by_flipping(images, angles)
print("New Number of images is: {}".format(len(augmented_images)))


# Split data into training and validation sets

X_train, X_validation, y_train, y_validation = train_test_split(augmented_images, augmented_angles, test_size=0.2)
train_set = [X_train, y_train]
validation_set = [X_validation, y_validation]

print("Number of records for training: {}".format(len(train_set[0])))
print("Number of records for validation: {}".format(len(validation_set[0])))


# Generator function

def generator(sets, batch_size):
    num_sets = len(sets[0])
    while 1:
        shuffle(sets)
        for offset in range(0, num_sets, batch_size):

            images = sets[0][offset:offset+batch_size]
            angles = sets[1][offset:offset+batch_size]
            X = np.array(images)
            y = np.array(angles)

            yield (X, y)

# Build Model

model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60, 20),(0, 0))))
model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Run Model

batch_size = 32
number_epochs = 5
train_generator = generator(train_set, batch_size)
validation_generator = generator(validation_set, batch_size)

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, steps_per_epoch=len(X_train) / batch_size,
                              validation_data=validation_generator,
                              validation_steps=len(X_train)/batch_size, epochs=number_epochs)

# Save Model
model.save('model.h5')

