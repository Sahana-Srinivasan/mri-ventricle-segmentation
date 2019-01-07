#LOAD IN DATA, DISPLAY EXAMPLE 3D IMAGES

import os
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters
from keras.utils import plot_model
import h5py
#cell to read in data from h5py and train the neural network
import matplotlib.pyplot as plt
%matplotlib inline
from ipywidgets import interact, FloatSlider
from skimage.segmentation import slic

from __future__ import print_function

import SimpleITK as sitk

#parameters for the cnn
batch_size = 1
num_classes = 2
epochs = 12
#img_rows, img_cols = 28, 28


# myshow and myshow3d obtained from https://simpleitk-prototype.readthedocs.io/en/latest/user_guide/visualization/plot_vis3dimage.html
def myshow(img):
    nda = sitk.GetArrayViewFromImage(img)
    plt.imshow(nda)
    
def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
        
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]
    
    elif nda.ndim == 4:
        c = nda.shape[-1]
        
        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")
            
        # take a z-slice
        nda = nda[nda.shape[0]//2,:,:,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]
    print ('xsize: ', xsize, 'ysize: ', ysize)
      
    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    
    t = ax.imshow(nda,extent=extent,interpolation=None)
    
    if nda.ndim == 2:
        t.set_cmap("gray")
    
    if(title):
        plt.title(title)
        
def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s,:,:] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[:,:,s] for s in zslices]
    
    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))
    
        
    img_null = sitk.Image([0,0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())
    
    img_slices = []
    d = 0
    
    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1
        
    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1
     
    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1
    
    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen,d])
        else:
            img_comps = []
            for i in range(0,img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
            img = sitk.Compose(img_comps)
            
    
    myshow(img, title, margin, dpi)
    
# Display example images
img = sitk.ReadImage('/Users/sahana/mri/patient1/bf_bf_ref_T1.nrrd')
img_labels = sitk.ReadImage('/Users/sahana/mri/patient1/VENT.nrrd')
img_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)

myshow3d(img)
size = img.GetSize()
myshow3d(img,yslices=range(50,size[1]-1,20), zslices=range(50,size[2]-1,20), dpi=100)
myshow3d(img_labels)
myshow3d(img_labels,yslices=range(50,size[1]-1,20), zslices=range(50,size[2]-1,20), dpi=100)


#print(img.GetWidth())
#print(img.GetHeight())
#print(img.GetDepth())
imarr1 = sitk.GetArrayFromImage(img)
imarr_labels1 = sitk.GetArrayFromImage(img_labels)

#segmentation = slic(img, n_segments = 3000, sigma = 5, slic_zero = 2)

#print ('slic segmentation image:')
#myshow3d(segmentation)

img2 = sitk.ReadImage('/Users/sahana/mri/patient2/bf_bf_ref_T1.nrrd')
img_labels2 = sitk.ReadImage('/Users/sahana/mri/patient2/VENT.nrrd')
imarr2 = sitk.GetArrayFromImage(img2)
imarr_labels2 = sitk.GetArrayFromImage(img_labels2)

img3 = sitk.ReadImage('/Users/sahana/mri/patient3/bf_bf_ref_T1.nrrd')
img_labels3 = sitk.ReadImage('/Users/sahana/mri/patient3/VENT.nrrd')
imarr3 = sitk.GetArrayFromImage(img3)
imarr_labels3 = sitk.GetArrayFromImage(img_labels3)

img4 = sitk.ReadImage('/Users/sahana/mri/patient4/bf_bf_ref_T1.nrrd')
img_labels4 = sitk.ReadImage('/Users/sahana/mri/patient4/VENT.nrrd')
imarr4 = sitk.GetArrayFromImage(img4)
imarr_labels4 = sitk.GetArrayFromImage(img_labels4)



imarr_data = np.array([imarr1,
                    imarr2,
                    imarr3,
                    imarr4])

imarr_labels = np.array([imarr_labels1,
                        imarr_labels2,
                        imarr_labels3,
                        imarr_labels4])

#imarr_data = np.concatenate((imarr1, imarr2, imarr3, imarr4), axis=3)
#imarr_labels = np.concatenate((imarr_labels1, imarr_labels2, imarr_labels3, imarr_labels4), axis=3)
#print(img_arr.shape)
#print(img_arr)

print (imarr_data.dtype, 'imgarr type')
print(imarr_labels.dtype, 'imarr_labels type')
print (imarr_data.shape)
print (imarr_labels.shape)
#counter = 0

#for i in range(len(img_arr)) :
#    print (i)
#    for j in range(len(img_arr[i])) :
#        for k in range(len(img_arr[i][j])) :
#            #print (i, j, k, len(img_arr[j]-1))
#            if img_arr[i][j][k] != 0.0:
#                counter +=1
#print (counter)

#defining CNN input shape
if K.image_data_format() == 'channels_first':
    imarr_data = imarr_data.reshape(4, 1, 512, 512, 192)
    imar_labels = imarr_labels.reshape(4,1,512,512,192)
    input_shape = (1, 512, 512, 192)
else:
    imarr_data = imarr_data.reshape(4, 512, 512, 192, 1)
    imarr_labels = imarr_labels.reshape(4,512,512,192,1)
    input_shape = (512, 512, 192, 1)

# store training data in the h5py file format
trainingFile = '/users/sahana/mri/TrainingData.hdf5'
print('creating ',trainingFile)
trainingDir = '/users/sahana/mri'
if not os.path.isdir(trainingDir):
    os.makedirs(trainingDir)
with h5py.File(trainingFile,'w') as patchFile:
    _patches = patchFile.create_dataset('data', data = imarr_data,dtype='float32',compression="gzip")
    _labels = patchFile.create_dataset('labels', data = imarr_labels,dtype='uint8',compression="gzip")

