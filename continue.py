"""
continue training an image classifier

Ian Zurutuza
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

FLAGS = None

import time

import tensorflow.keras.backend as K

def get_and_gen_images(module):
    """
    get images from image directory or url

    :param module: module (to get required image size info
    :return: batched image data
    """
    data_name = os.path.splitext(os.path.basename(FLAGS.image_dir_or_url))[0]
    print("data: ", data_name)

    # download images to cache if not already
    if FLAGS.image_dir_or_url.startswith('https://'):
        data_root = tf.keras.utils.get_file(data_name,
                                            FLAGS.image_dir_or_url,
                                            untar=True,
                                            cache_dir=os.getcwd())
    else:   # specify directory with images
        data_root = tf.keras.utils.get_file(data_name,
                                            FLAGS.image_dir_or_url)

    # image_size = hub.get_expected_image_size(module)

    # get image size for specific module
    image_size = module.input_shape

    print(image_size)

    # TODO: this is where to add noise, rotations, shifts, etc.
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    # create image stream
    train_image_data = image_generator.flow_from_directory(str(data_root),
                                                           target_size=(299, 299),
                                                           batch_size=FLAGS.batch_size,
                                                           subset='training')

    validation_image_data = image_generator.flow_from_directory(str(data_root),
                                                                target_size=(299, 299),
                                                                batch_size=FLAGS.batch_size,
                                                                subset='validation')

    return train_image_data, validation_image_data


def main():
    model = tf.keras.models.load_model(os.getcwd() + "/saved_models/modelname.h5")

    model.summary()


if __name__ == '__main__':
    main()
