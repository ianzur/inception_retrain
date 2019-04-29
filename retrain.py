"""
Retrain an image classifier

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


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])


def ensure_dir_exists(dir_name: str):
    """
    Makes directory if it doesn't exist

    :param dir_name: path to directory
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_imagenet_labels(predicted_labels: []):
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt',
                                          cache_dir=os.getcwd())

    imagenet_labels = np.array(open(labels_path).read().splitlines())

    readable_labels_batch = imagenet_labels[np.argmax(predicted_labels, axis=-1)]

    return readable_labels_batch


def decode_labels(predicted_labels: []):
    return


def decode_flower_labels(image_data, batch):
    label_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    label_names = np.array([key.title() for key, value in label_names])

    labels_batch = label_names[np.argmax(batch, axis=-1)]

    return labels_batch


#    TODO: test get from my directory
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


def show(images, labels, actual):
    plt.figure(figsize=(10, 9))

    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(images[n])
        plt.title(labels[n] + ' ' + actual[n])
        plt.axis('off')
        _ = plt.suptitle("ImageNet predictions")

    plt.show()


def create_model(bm, img_data):


    # model = tf.keras.Sequential()
    #
    # for layer in bm.layers[:-1]:
    #     model.add(layer)
    #


    # make pre-trained layers un-trainable
    for layer in bm.layers[:-1]:
        layer.trainable = False

    bm.summary()

    # inception_layer = tf.keras.layers.Lambda(bm, input_shape=img_data.image_shape, name='inception_v3')
    #
    # inception_layer.trainable = False

    # add pooling layer if you don't instantiate with pooling flag
    model = tf.keras.Sequential([
        bm,
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(img_data.num_classes, activation='softmax')
    ], name='my_model')

    # print(model.__name__)

    # compile model with optimizer adam
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def eval_checkpoints(module, validation):
    print("evaluating")
    print("new untrained model")

    init = tf.global_variables_initializer()

    with K.get_session() as sess:
        sess.run(init)
        model = create_model(module, validation)

        print(model.input_names)

        loss, acc = model.evaluate(validation)
        print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        # export_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

    # print(export_path)
    # export_model(module)


def main(_):
    # base_module = hub.Module(FLAGS.tfhub_module, name='inception_v3')


    input_tensor = tf.keras.layers.Input(shape=(299, 299, 3))

    # maybe this way?
    base_module = tf.keras.applications.inception_v3.InceptionV3(input_tensor=input_tensor,
                                                                 weights='imagenet',
                                                                 include_top=False,
                                                                 pooling='avg',     # or 'max',
                                                                 )

    # generate image stream
    train_image_data, validation_image_data = get_and_gen_images(base_module)

    # create checkpoint directory
    ensure_dir_exists(FLAGS.ckpt_dir)
    ckpt_path = FLAGS.ckpt_dir + "/retrain_ckpt"

    model = create_model(base_module, train_image_data)
    model.summary()

    print(model.input_shape)

    file = FLAGS.saved_model_dir + "/modelname.h5"

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    print(train_image_data.image_shape)

    sess = K.get_session()

    sess.run(init)

    steps_per_epoch = train_image_data.samples // train_image_data.batch_size
    batch_stats = CollectBatchStats()

    for epoch in range(1, FLAGS.epochs+1):
        print("\nepoch: {} / {}".format(epoch, FLAGS.epochs))

        model.fit((item for item in train_image_data),
                  steps_per_epoch=int(steps_per_epoch/10),
                  # validation_data=validation_image_data/10),
                  callbacks=[batch_stats])

        # save checkpoint
        saver.save(sess, ckpt_path, global_step=steps_per_epoch*epoch)

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)

    plt.show()

    model.save(file)

    model.summary()

    del model

    model = tf.keras.models.load_model(file)

    model.summary()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tfhub_module',
        type=str,
        default='/home/raphy/projects/vmi/gpu/pretrained_models/11d9faf945d073033780fd924b2b09ff42155763',
        help="""\
         Which TensorFlow Hub module to use. For more options,
         search https://tfhub.dev for image feature vector modules.\
         """)
    parser.add_argument(
        '--image_dir_or_url',
        type=str,
        default=(
            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'),
        help="""\
         What images to retrain on? 
         google "tensorflow example images", or make your own dataset.       
         see "label_video.py" for labeling video files
        """)
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'saved_models'),
        help='Where to save the exported graph.')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default=os.getcwd() + '/checkpoints',
        help='Where to save checkpoint files.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Where to save the exported graph.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default=os.getcwd() + '/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )


    FLAGS, unparsed = parser.parse_known_args()
    print([sys.argv[0]])

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
