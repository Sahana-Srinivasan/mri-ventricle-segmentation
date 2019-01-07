#TRAINING CNN (SMALL ONE) LINEAR 200 EPOCHS

#import libraries
import os
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, BatchNormalization
from keras import backend as K
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
#from vis.utils import utils
#from vis.utils.vggnet import VGG16
#from vis.visualization import visualize_activation, get_num_filters
from keras.utils import plot_model
import h5py
import matplotlib.pyplot as plt
%matplotlib inline
from ipywidgets import interact, FloatSlider
from __future__ import print_function
import SimpleITK as sitk
from keras import optimizers
from sys import platform as _platform 

#parameters for the cnn
batch_size = 128
num_classes = 1
epochs = 200
patch_size = 16


if (_platform == "darwin"):    
    trainingFile = '/users/sahana/mri/patches_size12_trialjuly22.hdf5'
    trainingDir = '/users/sahana/mri'
else:
    trainingFile = 'C:\\users\\sahana\\mri\\trainingdata_linearlabels.hdf5'
    trainingDir = 'C:\\users
    # read training data out of the h5py file and into an np array
with h5py.File(trainingFile,'r') as patchFile:
    images_tra = patchFile.get('data')[:]
    labels_tr = patchFile.get('labels')[:]
    
#images_tra = images_tr[0:24000,:,:,:]
#labels_tra = labels_tr[0:22000]

#labels_tra = keras.utils.to_categorical(labels_tr, num_classes)

print (images_tra.shape,'and labels: ', labels_tra.shape)

# convert the array of labels binary class matrices to fit the keras cnn's parameters
#labels_tr = keras.utils.to_categorical(labels_tr, num_classes)

#defining CNN input shape and re-sizing input arrays to match CNN's specifications
if K.image_data_format() == 'channels_first':
    images_tra = images_tra.reshape(images_tra.shape[0], 1, patch_size, patch_size, patch_size)
    input_shape = (1, patch_size, patch_size, patch_size)
else:
    images_tra = images_tra.reshape(images_tra.shape[0], patch_size, patch_size, patch_size, 1)
    input_shape = (patch_size, patch_size, patch_size, 1)
    
    

#design the cnn
model2 = Sequential()
model2.add(Conv3D(64,(3,3,3), activation='relu', name='conv1',input_shape=input_shape))
model2.add(MaxPooling3D((2,2,2), strides=(2,2,2)))
#model2.add(ZeroPadding3D((1,1,1)))
model2.add(Conv3D(32,(3,3,3), activation='relu', name='conv2'))
#model2.add(ZeroPadding3D((1,1,1)))
model2.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

#model.add(Conv3D(128, 3,3, 3, activation='relu',name='conv3'))
#model.add(ZeroPadding3D((1,1,1)))
#model.add(Conv3D(128, 3,3, 3, activation='relu',name='conv4'))
#model.add(ZeroPadding3D((1,1,1)))
#model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))


#model.add(ZeroPadding3D((2,2,2)))
#model.add(Conv3D(16, 4,4, 4, activation='relu',name='conv2'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

#model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))
#model.add(Conv3D(32, 4,4, 4, activation='relu',name='conv3'))
#model.add(ZeroPadding3D((2,2,2)))
#model.add(Conv3D(16, 2,2, 2, activation='relu',name='conv4'))
#model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))






model2.add(Flatten(name="flatten"))
#model.add(Dense(64, activation='relu',name='dense1'))

model2.add(Dense(256, activation='relu', name='dense1'))
model2.add(Dense(num_classes, activation='softmax', name='dense2'))

sgd = optimizers.SGD(lr=0.01)
model2.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.001),
              metrics=['accuracy'])

model2.summary()

#plot_model(model, to_file='model.png')

#train and test the model
history2 = model2.fit(images_tra, labels_tra,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.25)

#save the model
model2.save('segment_cnn1_linear_200.h5')

# plot training and validation accuracy
plt.subplot(1,2,1)
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plot training and validation loss on a separate graph
plt.subplot(1,2,2)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
axes = plt.gca()
#axes.set_xlim([0,5])
#axes.set_ylim([0,1])
plt.show()

#calculate test loss
#score = model.evaluate(images_te, labels_te, verbose=0)
#print('Test loss:', score[0])

