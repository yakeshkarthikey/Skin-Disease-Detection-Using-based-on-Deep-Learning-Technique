# %%
import numpy as np  
import pandas as pd

import os
print(os.listdir('D:/Py/aravindh_proj/dataset'))

# %% [markdown]
# **Import Libraries**

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os 
import tensorflow as tf
import sys
print(tf.keras.__version__)
from PIL import * 
from keras import optimizers
from keras import applications
from keras import backend as K
from os import listdir, makedirs
from keras.utils.data_utils import Sequence
from os.path import join, exists, expanduser
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam, SGD  
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from tensorflow.keras.applications import VGG16,ResNet101
from tensorflow.keras.utils import plot_model

# %% [markdown]
# **Function to display random images from each class of the dataset**

# %%
epochs = 15
path = 'D:/Py/aravindh_proj/dataset/'
classes = os.listdir(path)

def display_four_class_images(random_number):
    for i in classes:
        new_path = path + i
        random_image = os.listdir(new_path)[random_number]
        print(new_path + '/' +random_image)
        im = cv2.imread(new_path + '/' + random_image)[:,:,::-1]
        print(im)
        print(im.shape,im.dtype)
        plt.imshow(im)
        plt.show()

# %%
display_four_class_images(100)

# %%
def dataset_size(path, classes):
    size = []
    for i in classes:
        size.append(len(os.listdir(path + i)))
        
    df = pd.DataFrame(columns = ['Type', 'No_of_Images'])
    df['Type'] = classes
    df['No_of_Images'] = size
    
    return df

# %%
dataset_size(path, classes)

# %%
def average_image_size(label_name):
    r, g, b = [], [], []
    for image in os.listdir(path + label_name):
        im = cv2.imread(path + label_name+'/'+image)
        r.append(im.shape[0])
        g.append(im.shape[1])
        b.append(im.shape[2])
        
    return (sum(r)/len(r), sum(g)/len(g), sum(b)/len(b))

# %%
for label in classes:
    print(label, average_image_size(label))

# %%
def image_extensions(label_name):
    extension = []
    for image in os.listdir(path + label_name):
        extension.append(image.split('.')[-1])
        
    return list(set(extension))

# %%
for label in classes:
    print(label, image_extensions(label))

# %%
train_data_generator =  ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest',
                                validation_split = 0.25)

# %%


# %%
# img = load_img(path+'/Blight/Corn_Blight (412).JPG')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array  
# x = x.reshape((1,) + x.shape)  # this is a Numpy array  

# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in train_data_generator.flow(x, batch_size=1,
#                           save_to_dir='D:/Py/Bala_proj/corn or maize leaf disease dataset/', save_prefix='Corn_Blight', save_format='JPG'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

# %% [markdown]
# 

# %%
train_dataset = train_data_generator.flow_from_directory(
                                    path,
                                    target_size = (128, 128),
                                    class_mode = "categorical",
                                    batch_size = 16,
                                    subset = "training")

val_dataset = train_data_generator.flow_from_directory(
                                    path,
                                    target_size = (128, 128),
                                    class_mode = "categorical",
                                    batch_size = 16,
                                    subset = "validation")

# %%
def visualize_datagenerator(no_of_images):
    for pic in range(no_of_images):
        image, label = val_dataset.next()
        print("image shape is: ", image.shape)
        plt.imshow(image[0])
        print(image[20].shape)
        plt.show()

# %%
input_layer = Input(shape = (128, 128, 3))

vgg_model = VGG16(include_top=False,weights="imagenet",classes=train_dataset.num_classes, input_shape = (128, 128, 3))

vgg16_model = Sequential()
vgg16_model.add(vgg_model)

vgg16_model.add(Flatten())
vgg16_model.add(Dense(16,activation='relu'))
vgg16_model.add(Dropout(0.25))
vgg16_model.add(Dense(4,activation='softmax'))

adam = Adam(learning_rate= 0.0001, decay=0.0001 / epochs)


# %%
vgg16_model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ["accuracy"])

# %%
vgg16_model.summary()

# %%
history = vgg16_model.fit(train_dataset, epochs = epochs, validation_data = val_dataset, verbose = True)

# %%
history

score = vgg16_model.evaluate(val_dataset)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %%
print(history.history["accuracy"])

# %%
def visualize_training_epochs(v):
    plt.plot(v.history["accuracy"])
    plt.plot(v.history["val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("vgg16_model_accuracy")
    plt.legend(["Train", "Validation"])
    plt.show()
    
    plt.plot(v.history["loss"])
    plt.plot(v.history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("vgg16_model_loss")
    plt.legend(["Train", "Validation"])
    plt.show()

# %%
visualize_training_epochs(history)

# %%
vgg16_model.save("D:/Py/aravindh_proj/model-1.h5")

# %%
print(val_dataset.class_indices)

# %%
print(val_dataset.classes)

# %%
print(len(val_dataset))

# %%
predictions = vgg16_model.predict(val_dataset)
print(predictions)

# %%
prediction_class = np.argmax(predictions, axis = 1)
prediction_class

# %%
print(len(prediction_class))

# %%
print(confusion_matrix(val_dataset.classes, prediction_class))

# %%
print(classification_report(val_dataset.classes, prediction_class))

# %%



