import io
import os

import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import argparse
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, \
    Dropout, ReLU, Conv2DTranspose, Dense, Flatten, Input, Concatenate
from tensorflow import keras
from tools import *

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Стандартные пути
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangul.txt')
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model/')

DEFAULT_LABEL_CSV_FILE = os.path.join(SCRIPT_PATH, './image-data/labals-map.csv')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-images/')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, './output-tfrecords/')

MODEL_NAME = 'hangul-localizator'

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
FRAMES = 10


def load_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)/255
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def parse_record(record):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'coords': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, feature_description)
    image = tf.io.parse_tensor(parsed_record['image'], out_type=tf.float32)
    feature = tf.io.parse_tensor(parsed_record['coords'], out_type=tf.float32)
    return image, feature


def get_dataset(tfrecord_dir):
    print("Создание датасета...")
    train_pattern = sorted(glob.glob(tfrecord_dir + 'train-*.tfrecords'))
    test_pattern = sorted(glob.glob(tfrecord_dir + 'test-*.tfrecords'))

    train_dataset = tf.data.TFRecordDataset(train_pattern) \
        .map(parse_record) \
        .shuffle(1000) \
        .batch(BATCH_SIZE) \
        .prefetch(1)
        # .repeat(2) \

    test_dataset = tf.data.TFRecordDataset(test_pattern) \
        .map(parse_record) \
        .batch(BATCH_SIZE) \
        .prefetch(1)
    print("Датасет создан...")
    return train_dataset, test_dataset


def create_model(frames):
    inputs = Input((IMAGE_SIZE + (3,)))
    x = Conv2D(32,  3, activation='relu', padding='same')(inputs)
    x = Conv2D(64,  3, activation='relu', padding='same', strides = 2)(x)
    x = Conv2D(64,  3, activation='relu', padding='same')(x)
    x = Conv2D(64,  3, activation='relu', padding='same', strides = 2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same', strides = 2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same', strides = 2)(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(4 * frames)(x)
    return Model(inputs, x, name=MODEL_NAME)


class LocalizationModel(tf.keras.Model):
    def __init__(self, model, input_width, input_height, frames=FRAMES):
        super(LocalizationModel, self).__init__()
        self.frames = frames
        self.input_width = input_width
        self.input_height = input_height
        self.model = model
        
        self.box_optimizer = tf.keras.optimizers.Adam(1e-4, clipnorm=1)
    
    @tf.function
    def training_step(self, x, true_boxes):
        with tf.GradientTape() as tape_box:
            pred = self.model(x, training=True)
            pred = tf.reshape(pred, [-1, self.frames, 4])
            loss = IoU_Loss(true_boxes, pred, frames=self.frames)
        # Backpropagation.
        grads = tape_box.gradient(loss, self.model.trainable_variables)
        self.box_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def normalize(self, coords_tensor):
        width = self.input_width
        height = self.input_height
        normalized_tensor = coords_tensor / [width, height, width, height] *2-1
        return normalized_tensor
    
    def nn_fit(self, x, epochs):
        hist = np.array(np.empty([0]))
        num_elements = 0
        for epoch in range(1, epochs + 1):
            loss, loss_count = 0, 0
            
            for step, (i, c) in enumerate(x):
                normalized = self.normalize(c)
                loss += tf.reduce_mean(self.training_step(i, normalized))
                loss_count += 1
                if epoch == 1:
                    num_elements = step + 1
                print(f"Epoch: {epoch}/{epochs} | {step+1:_}/{num_elements:_} | loss: {loss/loss_count:.4f}", end="\r")
            hist = np.append(hist, loss/loss_count)
            
            model_chkpt_step_path = os.path.join(DEFAULT_MODEL_DIR, f"epoch{epoch}-{MODEL_NAME}.h5")
            self.model.save(model_chkpt_step_path)
        return hist
    
    def testing(self, dataset, sourse_frame=False):
        """ Проверка работы """
        width = self.input_width
        height = self.input_height
        
        for ii, cc in dataset.take(1):
            #обрабатывем целый батч, используем только пять элементов
            pred = self.model(ii)
            plt.figure(figsize=(10, 6))
            print(pred)
            
            for num in range(3):
                pred = tf.reshape(pred, [-1, self.frames, 4])
                
                ax = plt.subplot(1, 5, num+1)
                #переход в numpy для работы в opencv
                i = ii[num].numpy()
                c = pred[num].numpy()
                # c = (c+1)/2*128 #обратно из от -1...1 к 0...64
                # делать денормализацию 
                c = (c+1)/2 * [width, height, width, height]
                c = c.astype(np.int16)  #для opencv
                print(c)
                # предсказанные рамки
                for bb in c:
                    xy1 = (int(min(bb[0], bb[2])), int(bb[1]))
                    xy2 = (int(max(bb[0], bb[2])), int(bb[3]))
                    i = cv2.rectangle(i, xy1, xy2, (0,1,0), 1)
                plt.imshow(i)
            plt.show()


def main(tfrecord_dir, use_checkpoint=1):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    
    model_save_path = os.path.join(DEFAULT_MODEL_DIR, f"{MODEL_NAME}.h5")

    # Создание датасета
    train_dataset, test_dataset = get_dataset(tfrecord_dir)
        
    localizer = create_model(frames=FRAMES)
    model = LocalizationModel(model=localizer, input_width=IMAGE_SIZE[0], input_height=IMAGE_SIZE[1])
    
    # model.testing(train_dataset, sourse_frame=False)
    
    # пример работы 
    # for i, c in train_dataset.take(1):
        # print(c.shape, end='\n')
        # print(c, end='\n')
        # print(model.normalize(c))
        # print(tf.reduce_mean(model.training_step(i, c)))
        # print(tf.reduce_mean(model.training_step(i, model.normalize(c))))

    history = model.nn_fit(train_dataset, epochs=30)
    plt.plot(np.arange(0, len(history)), history)
    plt.show()
    model.testing(test_dataset)
    
    model.model.save(f"{model_save_path}")
    
    # model.model.load_weights(f"{MODEL_NAME}.h5")
    # model.testing(test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord-dir', type=str, dest='tfrecord_dir', default=DEFAULT_TFRECORDS_DIR)
    args = parser.parse_args()
    main(args.tfrecord_dir)
