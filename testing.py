#GENERATE PREDICTION, NRRD, AND EVALUATE
import h5py
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
from keras.utils import plot_model
from keras import applications
from scipy.misc import imsave
import time
from keras.applications import vgg16
from keras.models import load_model
import SimpleITK as sitk
from sys import platform as _platform 
import nrrd

def generate3DTestPatchesBinary(mriNo, outputFile, channel, patchSize, cnn):
    patches_chunk = []
    labels_chunk= []
    prediction_chunk = []
    #startTestNo = 25
    #endTestNo = 31
    patch_size = 16

    if (_platform =="darwin"):
        img = sitk.ReadImage('/Users/sahana/mri/patient' + str(mriNo) + '/bf_bf_ref_' + channel + '.nrrd')
        img_gt = sitk.ReadImage('/Users/sahana/mri/patient' + str(mriNo) + '/' + 'VENT.nrrd')
        img_bm = sitk.ReadImage('/Users/sahana/mri/patient' + str(mriNo) + '/BrainMask_robexLST.nrrd')
    else:
        img = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(mriNo) + '\\bf_bf_ref_T1.nrrd')
        img_gt = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(mriNo) + '\\VENT.nrrd')
        img_bm = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(mriNo) + '\\BrainMask_robexLST.nrrd')
        
    if K.image_data_format() == 'channels_first':
        #images_tra = images_tra.reshape(images_tra.shape[0], 1, patch_size, patch_size, patch_size)
        input_shape = (1000, 1, patch_size, patch_size, patch_size)
    else:
        #images_tra = images_tra.reshape(images_tra.shape[0], patch_size, patch_size, patch_size, 1)
        input_shape = (1000, patch_size, patch_size, patch_size, 1)
        
    # load in images
    imarr = sitk.GetArrayFromImage(img)
    imarr_gt = sitk.GetArrayFromImage(img_gt)
    imarr_bm = sitk.GetArrayFromImage(img_bm)
    imarr_max, imarr_min = np.amax(imarr), np.amin(imarr)
    imarr_norm = np.multiply(imarr,imarr_bm)
    imarr_norm = 2*((imarr_norm-imarr_min)/(imarr_max-imarr_min))-1
    counter = 0
    
    patchFile = h5py.File(outputFile, 'w')
    predict_labels = patchFile.create_dataset('prediction', (1000,2),maxshape=(None,2),
                                       chunks=(1000,2), dtype='float32')

    # load in model
    model = load_model('segment_cnn' + str(cnn) + '_binary_200.h5')

    # converte test images into proper patch format to feed into model
    for i in range(int(patchSize/2),512-int(patchSize/2)):
            for j in range(int(patchSize/2),512-int(patchSize/2)):
                for k in range(int(patchSize/2),192-int(patchSize/2)):
                    if (imarr_bm[i][j][k])==1:
                           
                        #print ('hi')
                        val = np.float32(0)
                        x1 = int(i-(patchSize/2))
                        y1 = int(j-(patchSize/2))
                        z1 = int(k-(patchSize/2))

                        patch = imarr_norm[x1:x1+patchSize,y1:y1+patchSize,z1:z1+patchSize]
                        #val = imarr_gt[i][j][k]
                        patches_chunk.append(patch)
                        counter += 1
                        if (counter%1000000 == 0):
                            print (str(mriNo), counter)
                        if (counter== 1000):
                            #print ('hello')
                            #patches[:] = patches_chunk
                            
                            input_data = np.asarray(patches_chunk)
                            input_data = input_data.reshape(input_shape)
                            prediction_chunk = model.predict(input_data)
                            #labels[:] = labels_chunk
                            predict_labels[:] = prediction_chunk
                            patches_chunk = []
                            #labels_chunk = []
                            prediction_chunk = []
                            predict_labels.resize(predict_labels.shape[0]+1000, axis=0)
                            #patches.resize(patches.shape[0]+1000, axis=0)
                            #labels.resize(labels.shape[0]+1000, axis=0) 
                            #print (counter, patches.shape,labels.shape)
                        else:
                            if (counter%1000 == 0):
                                #print ('hello')
                                #patches[(counter-1000):] = patches_chunk
                                #labels[(counter-1000):] = labels_chunk
                                input_data = np.asarray(patches_chunk)
                                input_data = input_data.reshape(input_shape)
                                prediction_chunk = model.predict(input_data)
                                patches_chunk = []
                                #labels_chunk = []
                                predict_labels[(counter-1000):] = prediction_chunk
                                prediction_chunk = []
                                
                                predict_labels.resize(predict_labels.shape[0]+1000, axis=0)
                                #labels.resize(labels.shape[0]+1000, axis=0) 
                                #print (counter, patches.shape, labels.shape)
 
    print (counter)
    

    #patches.resize(counter, axis=0)
    #labels.resize(counter,axis=0)
    predict_labels.resize(counter,axis=0)
    
    #patches[(counter-(counter%1000)):] = patches_chunk
    #labels[(counter-(counter%1000)):] = labels_chunk
    
    if K.image_data_format() == 'channels_first':
        #images_tra = images_tra.reshape(images_tra.shape[0], 1, patch_size, patch_size, patch_size)
        input_shape2 = ((counter%1000), 1, patch_size, patch_size, patch_size)
    else:
        #images_tra = images_tra.reshape(images_tra.shape[0], patch_size, patch_size, patch_size, 1)
        input_shape2 = ((counter%1000), patch_size, patch_size, patch_size, 1)
    
    #nput_data = patches_chunk.reshape(input_shape2)
    input_data = np.asarray(patches_chunk)
    input_data = input_data.reshape(input_shape2)
    prediction_chunk = model.predict(input_data)
    predict_labels[(counter-(counter%1000)):] = prediction_chunk
    patchFile.close()
    
    
#define main variables
patchSize = int(16)
#numPatch = 1000
#numPatient = 24
channel = 'T1'
#model_pred = load_model('segment_model_linearlabels.h5')
#print ('Model loaded.')
cnn = 1

#for pat in range(25,31):
#    if (_platform =="darwin"):
#        outputFile = '/users/sahana/mri/patchsize16_test_patient25_binarylabels.hdf5'
#    else:
#        outputFile = 'C:\\users\\sahana\\mri\\patient' + str(pat) + '_prediction_binary' + str(cnn) + '.hdf5'
#
#    #predictFile = '/users/sahana/mri/predict_test1_patient25.hdf5'
#    print ('Generating patches for patient ' + str(pat))
#    generate3DTestPatchesBinary(pat, outputFile, channel, patchSize, 1) 
#    print ('Patches and prediction generated; exercise completed for patient' + str(pat))  
    
def generateNRRD(patientNo, labelType, cnnNo):
    print ('patient no' + str(patientNo))
    img_gt = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\VENT.nrrd')
    imarr_gt = sitk.GetArrayFromImage(img_gt)
    img_bm = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\BrainMask_robexLST.nrrd')
    imarr_bm = sitk.GetArrayFromImage(img_bm)
    imarr_predict = np.zeros((imarr_bm.shape[0],imarr_bm.shape[1],imarr_bm.shape[2]))
    fileName = 'C:\\users\\sahana\\mri\\patient' + str(patientNo)+ '_predict_' + labelType + str(cnnNo) + '_.nrrd'
    fileName2 = 'C:\\users\\sahana\\mri\\patient' + str(patientNo)+ '_predict_' + labelType + str(cnnNo) + '.nrrd'
    
                            
    predictLoc = 'C:\\users\\sahana\\mri\\patient' + str(patientNo) + '_prediction_binary' + str(cnn) + '.hdf5'
    predictFile = h5py.File(predictLoc, 'r')
    counter = 0
    patchSize = 16              
            
                            
    for i in range(int(patchSize/2),512-int(patchSize/2)):
        for j in range(int(patchSize/2),512-int(patchSize/2)):
            for k in range(int(patchSize/2),192-int(patchSize/2)):
                if (imarr_bm[i][j][k]==1):
                       # vals = predictFile.get('labels')[counter]
                        imarr_predict[i][j][k] = np.argmax(predictFile.get('prediction')[counter])
                        if (counter%1000000==0):
                            print (counter)
                        counter+=1
    # save predictions as images
    print ('starting to save segmentation image')
    nrrd.write(fileName, imarr_predict)
    img_predict = sitk.ReadImage(fileName)
    imarr_predict2 = sitk.GetArrayFromImage(img_predict)
    nrrd.write(fileName2,imarr_predict2)
    print ('saved')
    predictFile.close()
    
#for pati in range(25, 31):
#    generateNRRD(pati, 'binary', 1)

def evaluateSegmentation(patientNo, labelType, cnnNo):
    patchSize = 16
    img_gt = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\VENT.nrrd')
    imarr_gt = sitk.GetArrayFromImage(img_gt)
    img_bm = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '\\BrainMask_robexLST.nrrd')
    imarr_bm = sitk.GetArrayFromImage(img_bm)
    img_predict = sitk.ReadImage('C:\\users\\sahana\\mri\\patient' + str(patientNo) + '_predict_' + labelType + str(cnnNo) + '.nrrd')
    imarr_predict = sitk.GetArrayFromImage(img_predict)
    print ('patient no' + str(patientNo))
    print ('size of predict nrrd ', imarr_predict.shape)
    #imarr_predict_norm = imarr_predict*20
    #fileName2 = 'C:\\users\\partha\\sahana\\patchsize16_patient25_predict_' + labelType + str(cnnNo) + '_norm.nrrd'
    #nrrd.write(fileName2, imarr_predict_norm)
    
    truePos, trueNeg, falsePos, falseNeg = 0,0,0,0
    pos, neg = 0,0
    #imarr_results_display = np.zeros((imarr_bm.shape[0],imarr_bm.shape[1],imarr_bm.shape[2]))
    print (np.unique(imarr_predict))

    for i in range(int(patchSize/2),512-int(patchSize/2)):
        for j in range(int(patchSize/2),512-int(patchSize/2)):
            for k in range(int(patchSize/2),192-int(patchSize/2)):
                if (imarr_bm[i][j][k])==1:
                        if (imarr_predict[i][j][k] >= 0.5):
                            pos +=1
                            if (imarr_gt[i][j][k] == 1):
                                truePos +=1
                            else:
                                falsePos +=1
                        else:
                            neg+=1
                            #imarr_results_display[i][j][k] = 0
                            if (imarr_gt[i][j][k] == 0):
                                trueNeg +=1
                            else:
                                falseNeg +=1
    print (pos, neg, truePos, falsePos, trueNeg, falseNeg)
    diceCoeff = 2*truePos/(2*truePos + falsePos + falseNeg)
    posRate = truePos/(truePos+falseNeg)
    #fileName1 = 'C:\\users\\partha\\sahana\\patchsize16_patient25_predict_' + labelType + str(cnnNo) + '_segmentmask.nrrd'

    print ('Dice coefficient: ' + str(diceCoeff))
    print ('True positive rate: ' + str(posRate))
    #nrrd.write(fileName1, imarr_results_display)
    
for patie in range(25, 31):
    evaluateSegmentation(patie, 'binary',1)
