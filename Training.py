import tensorflow as tf
from PIL import Image
#from google.colab import drive
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Model
import keras
from tensorflow.keras.applications import vgg19
import numpy as np
from tensorflow.keras.optimizers import SGD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from datetime import datetime
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules

Image.MAX_IMAGE_PIXELS = None

#style_transfer_url = 'Users/francescapunzetti/Desktop/colored.jpg'
#base_url ='Users/francescapunzetti/Desktop/bianco.jpg'
style_image_path = 'coloredmini.tif'
base_image_path = 'biancomini.tif'

# read the image file in a numpy array
a = plt.imread('coloredmini.tif')
b = plt.imread('biancomini.tif')
f, axarr = plt.subplots(1,2, figsize=(15,15))
axarr[0].imshow(a)
axarr[1].imshow(b)
#plt.show()

model = vgg19.VGG19(
    include_top=False,
    weights='imagenet',
)
# set training to False
model.trainable = False
# Print details of different layers

model.compile(optimizer='adam', metrics=['accuracy'])


model.summary()

def display_image(image):
    # remove one dimension if image has 4 dimension
    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
 
    img = deprocess_image(img)
 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return

def preprocess_image(image_path):
    # Function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):

    # Conversion into an array
    x = x.reshape((img_nrows, img_ncols, 3))
   
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Conversion from BGR to RGB.
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype("uint8")

    return x

#dimensions of the generated picture
width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

content_layer = "block5_conv2"
content_model = Model(
    inputs=model.input,
    outputs=model.get_layer(content_layer).output
)
content_model.summary()

style_layers = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv1",
]
style_models = [Model(inputs=model.input,
                      outputs=model.get_layer(layer).output) for layer in style_layers]


def content_loss(content, generated):
    return tf.reduce_sum(tf.square(generated - content))

def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


weight_of_layer = 1. / len(style_models)


def style_cost(style, generated):
    J_style = 0
 
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * weight_of_layer
 
    return J_style

generated_images = []

#preparing th model 

def result_saver(iteration):
  # Create name
  now = datetime.now()
  now = now.strftime("%Y%m%d_%H%M%S")
  model_name = 'best_model'
  image_name = str(iteration) + '_' + str(now)+"_image" + '.tif'

  # Save image
  img = deprocess_image(combination_image.numpy())
  keras.preprocessing.image.save_img(image_name, img)
  #model.save_weights('./best_model.t7')
  model.save('Desktop/2/best_model.h5')

content = preprocess_image(base_image_path)
style = preprocess_image(style_image_path)
generated = tf.Variable(content, dtype=tf.float32)
    
opt = tf.keras.optimizers.Adam(learning_rate=0.7)  

def train_step(generated_image):
    with tf.GradientTape() as tape:

        J_style = style_cost(style, generated)
        J_content = content_loss(content, generated)
        J_total = 10 * J_content + 1000 * J_style
        
    grads = tape.gradient(J_total, generated)
    opt.apply_gradients([(grads, generated)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
    print('Iteration Loss {:e}.'.format(J_total))
    epochs_loss_avg.update_state(J_total)  
    best_image = generated.numpy()
    # For grading purposes
    return J_total, best_image


# Train the model and get best image

epochs = 2
steps_per_epoch=5
step = 0
epochs_loss_avg = tf.keras.metrics.Mean()
epochs_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
best_image = None

for n in range(0, epochs +1):
    print(f"Epoch {n} ")
    for m in range(0, steps_per_epoch):
        print("Iteration :{}".format(m))
        step += 1
        model.load_weights('weight_of_layer.h5')  
        train_step(generated)


    #print('Total Loss {:e}.'.format(content_loss))
            #epoch_accuracy.update_state(style_image_path, best_image)
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(n,
                                                                epochs_loss_avg.result(),
                                                                epochs_accuracy.result()))
    model.save_weights('weight_of_layer.h5')
#final_img = best_image
result_saver(epochs)



# code to display best generated image and last 10 intermediate results
#plt.figure(figsize=(12, 12))
 
#for i in range(10):
   # plt.subplot(4, 3, i + 1)
   # display_image(generated_images[i])
#plt.show()
 
# plot best result

  
#display_image(final_img)
#plt.show()
newmodel= load_model('Desktop/2/best_model.h5')
newmodel.summary()
