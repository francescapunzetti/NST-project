# NST model for histopathological images
The aim of this project is to implement a neural style transfer network which can transfer the pattern from a histological slide stained with hematoxylin-eosin, to an other slide without any type of stain.

## Overview
The file in this repository are: 
* ``Training.py`` , where the model has been created and trained 
* ``neural_style_transfer.py`` , where the model already trained, is used on other images, to transfer the style in a quicker way

## Procedure
The first step is to obtain a neural network, using two images of the same sample in subsequent slides. One is stained and the other not; in this way the two sample are as similar much as possible, and the only difference between them is the presence of hematoxylin-eosin.

Obviously the first will be the style reference of our model.

(WORK IN PROGRESS)
