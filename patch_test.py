import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import *
import matplotlib.pyplot as plt


def crop_with_direction(data, left=0, right=0, bottom=0, top=0):
    # if(sum([left, right, bottom, top]) != 1):
    #     print("more than 1 direction was given")
    #     return "ERROR"

    cropped_data = tf.image.crop_to_bounding_box(data, top, left, 32-bottom, 32-right)
    padded_data = tf.image.pad_to_bounding_box(cropped_data, top, left, 32, 32)
    plt.imshow(padded_data)
    return padded_data


# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# img = x_train[0]

# prcsed=crop_with_direction(img,1,0,0,0)

print(1)
