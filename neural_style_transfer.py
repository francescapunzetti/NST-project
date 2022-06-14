# import the necessary packages
import argparse
#import imutils
import time
from tkinter import Y
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import Model
import matplotlib.pyplot as plt
import keras
import os 
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import decode_predictions
from keras.preprocessing import image
from tensorflow.python.keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img
#python neural_style_transfer.py --image /Users/francescapunzetti/Desktop/2/fre.jpg \   
#        --model /Users/francescapunzetti/Desktop/2/the_wave.t7
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import h5py
import numpy as np
import PIL.Image
from PIL import Image as im
import time
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

from keras.models import load_model
import cv2
import numpy as np

model = load_model('best_model.h5')

model.compile(optimizer='adam',
              metrics=['accuracy'])

img = cv2.imread('fre.jpg')
img = cv2.resize(img,(320,240))
img = np.reshape(img,[1,320,240,3])

classes = model.predict(img, batch_size =1)
print(classes.shape)

data= tensor_to_image(classes)
#data = im.fromarray(classes)
#if data.mode != 'RGB':
    #data = data.convert('L')
data.save('transformed.jpg')





#m = h5py.File('best_model.h5', 'r')
