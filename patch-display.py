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

def plotImage(data,x,y,z):
    #inputImage = sitk.ReadImage(imageFile)
    #data = sitk.GetArrayFromImage(inputImage)
    #img = sitk.GetImageFromArray(data)
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(18.5, 10.5)
    axs[0].imshow(data[x,:,:],interpolation="nearest", cmap = 'gray')
    axs[0].set_axis_off()
    axs[1].imshow(np.flipud(data[:,y,:]),interpolation="nearest", cmap = 'gray')
    axs[1].set_axis_off()
    axs[2].imshow(np.flipud(data[:,:,z]),interpolation="nearest", cmap = 'gray')
    axs[2].set_axis_off()


img = sitk.ReadImage('C:\\Users\\sahana\\mri\\patient25\\bf_bf_ref_T1.nrrd')
img_bm = sitk.ReadImage('C:\\Users\\sahana\\mri\\patient25\\BrainMask_robexLST.nrrd')
imarr = sitk.GetArrayFromImage(img)
imarr_bm = sitk.GetArrayFromImage(img_bm)
imarr_max, imarr_min = np.amax(imarr), np.amin(imarr)
imarr_norm = np.multiply(imarr,imarr_bm)
imarr_norm = 2*((imarr_norm-imarr_min)/(imarr_max-imarr_min))-1

#trainingFile = '/users/sahana/mri/patchsize16_floatlabels_july28'
patchSize = 16

#read training data out of the h5py file and into an np array
#with h5py.File(trainingFile,'r') as patchFile:
#    images_train = patchFile.get('data')[:]
#    labels_train = patchFile.get('labels')[:]

counter = 0
#display example training images (patches of the MRIs)
while (counter == 0):   
    img_ind_x = np.random.randint(patchSize/2,512-(patchSize/2))
    img_ind_y = np.random.randint(patchSize/2,512-(patchSize/2))
    img_ind_z = np.random.randint(patchSize/2,192-(patchSize/2))

    if (imarr_bm[img_ind_x][img_ind_y][img_ind_z] == 1):
        x1 = int(img_ind_x-(patchSize/2))
        y1 = int(img_ind_y-(patchSize/2))
        z1 = int(img_ind_z-(patchSize/2))
        patch_norm = imarr_norm[x1:x1+patchSize,y1:y1+patchSize,z1:z1+patchSize]
        plotImage(patch_norm,patchSize-1,patchSize-1,patchSize-1)
        counter+=1
