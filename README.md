# mri-ventricle-segmentation
Machine learning-based segmentation of lateral brain ventricles in MRIs using Keras

This contains one example pipeline of generating training "patches" of a 3D MRI,
labelling those patches as part of the lateral ventricle or not, desining a CNN
model in Keras, training it on aforementioned patches, testing the model on 
seprate MRIs, and evaluating the accuracy of the model's prediction of where
in the MRI the ventricles are. Other CNNs were desgined, trained, and tested but
their architectures are not shown here. Simple ITK, scikit-learn, Keras were all used.
