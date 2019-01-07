
# EXTRACT BINARY TRAINING PATCHES ON 24 IMAGES
import os
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras import backend as K
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

#reference for axes: x is sagittal, y is coronal, z is axial

# Generate patches to train CNN from (patches dervied from MRIs)
def generate3DTrainingPatches(npatch, channel, mriNumbers, patchSize, outputFile):
    patches = []
    labels= []
    for patientNo in range(1,mriNumbers+1):
        #set up for only one channel so far - T1
        img = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\bf_bf_ref_' + channel + '.nrrd')
        img_gt = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\VENT.nrrd')
        img_bm = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\BrainMask_robexLST.nrrd')
        imarr = sitk.GetArrayFromImage(img)
        imarr_gt = sitk.GetArrayFromImage(img_gt)
        imarr_bm = sitk.GetArrayFromImage(img_bm)
        
        imarr_max, imarr_min = np.amax(imarr), np.amin(imarr)
        imarr_norm = np.multiply(imarr,imarr_bm)
        imarr_norm = 2*((imarr_norm-imarr_min)/(imarr_max-imarr_min))-1

        numBorder, numNeg, numPos = 0, 0, 0

        # use random sample of patches from within MRI but with specific ratio of patches 
        # that are positively or negatively labelled as being part of the ventricle
        while (numNeg+numBorder+numPos<npatch):

            img_ind_x = np.random.randint(patchSize/2,512-(patchSize/2))
            img_ind_y = np.random.randint(patchSize/2,512-(patchSize/2))
            if patientNo !=18:
                img_ind_z = np.random.randint(patchSize/2,192-(patchSize/2))
            else:
                img_ind_z = np.random.randint(patchSize/2,176-(patchSize/2))
            if (imarr_bm[img_ind_x][img_ind_y][img_ind_z]==1):
                val = imarr_gt[img_ind_x][img_ind_y][img_ind_z]

                x1 = img_ind_x-(int(patchSize/2))
                y1 = img_ind_y-(int(patchSize/2))
                z1 = img_ind_z-(int(patchSize/2))

                patch = imarr_norm[x1:x1+patchSize,y1:y1+patchSize,z1:z1+patchSize]

                if val>0:
                    if numPos <.25*npatch:
                        patches.append(patch)
                        labels.append(val)
                        numPos+=1
                elif numBorder<.25*npatch or numNeg<.5*npatch:
                    posVox, negVox = 0, 0
                    for i in range(x1,x1+patchSize):
                        for j in range(y1,y1+patchSize):
                            for k in range(z1,z1+patchSize):
                                if imarr_gt[i][j][k] ==0:
                                    negVox+=1
                                else: 
                                    posVox+=1
                       
                    if negVox>0 and posVox>0:
                        if numBorder<.25*npatch:
                            numBorder+=1
                            patches.append(patch)
                            labels.append(val)

                    elif negVox>0:
                        if numNeg<.50*npatch:
                            numNeg+=1
                            patches.append(patch)
                            labels.append(val)   


        print ('Done with patient no. ' + str(patientNo))


    patches = np.array(patches)
    labels = np.array(labels)
    print ('patches', patches.shape)
    print ('labels', labels.shape)
        
    with h5py.File(outputFile,'w') as patchFile:
        _datashape = [patches.shape[0], patchSize, patchSize, patchSize]
        patches = patchFile.create_dataset('data', data=patches, shape=_datashape, dtype='float32')
        _labelshape = [patches.shape[0]]
        labels = patchFile.create_dataset('labels', data=labels, shape=_labelshape, dtype='uint8')
         
        
#define main variables
patchSize = 16
numPatch = 1000
numPatient = 24
channel = 'T1'
outputFile = 'C:\\users\\sahana\\mri\\trainingdata_binarylabels.hdf5'
#for mri in range(1,numPatient+1):
generate3DTrainingPatches(numPatch, channel, numPatient, patchSize, outputFile) 
#print ("WE ARE DONE WITH PATIENT NO." + str(mri))

# read test data out of h5py and into np array
with h5py.File(outputFile,'r') as patchFile:
    images_te = patchFile.get('data')[:]
    labels_te = patchFile.get('labels')[:]
    
print ('data array size read out of h5py: ', images_te.shape)
print ('labels array size read out of h5py: ', labels_te.shape)
print ('DONE WITH THE EXERCISE')
