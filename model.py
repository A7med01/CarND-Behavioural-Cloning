
import csv 
import matplotlib.image as mpimg
import numpy as np

lines = []

with open ('data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader :
        lines.append(line)
        

images = []
measurements = [] 

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = mpimg.imread(current_path)  
    images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
    
print(10001)

from keras.models import Sequential , Model
from keras.layers.core import Dense, Activation, Flatten, Dropout,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(6, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(16, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1)) 
model.compile(optimizer = 'adam', loss='mse')
model.fit(X_train, y_train, epochs=4, validation_split=0.2, shuffle = True)  
model.save('model1.h5')