import io
import os

import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import json
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, \
    Dropout, ReLU, Conv2DTranspose, Dense, Flatten, Input, Concatenate
from tensorflow import keras
from tools import IoU_Loss

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
DEFAULT_MASK_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-masks/')
# DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, './output-tfrecords/')
DEFAULT_TFRECORDS_DIR = "C:/Projects/tfrecords/"


MODEL_NAME = 'hangul-localizator'

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
FRAMES = 5


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


def get_dataset():
    print("Создание датасета...")
    train_pattern = sorted(glob.glob(DEFAULT_TFRECORDS_DIR + 'train-*.tfrecords'))
    test_pattern = sorted(glob.glob(DEFAULT_TFRECORDS_DIR + 'test-*.tfrecords'))

    train_dataset = tf.data.TFRecordDataset(train_pattern) \
        .map(parse_record) \
        .shuffle(1000) \
        .batch(BATCH_SIZE) # \
        # .prefetch(1)

    test_dataset = tf.data.TFRecordDataset(test_pattern) \
        .map(parse_record) \
        .batch(BATCH_SIZE) # \
        # .prefetch(1)
    print("Датасет создан...")
    return train_dataset, test_dataset


class LocalizationModel(tf.keras.Model):
    def __init__(self, frames=FRAMES):
        super(LocalizationModel, self).__init__()
        self.frames = frames
        self.nn_box = self.create_model(self.frames)
        
        self.box_optimizer = tf.keras.optimizers.Adam(1e-4, clipnorm = 1.0)
    
    @tf.function
    def training_step(self, x, true_boxes):
        with tf.GradientTape() as tape_box:
            print(x.shape)
            print(true_boxes.shape)
            
            pred = self.nn_box(x, training=True)
            # pred = tf.reshape(pred, [-1, self.frames, 3])
            print(x.shape)
            print(pred.shape)
            pred = tf.reshape(pred, [-1, self.frames, 4])
            print(x.shape)
            print(pred.shape)
            
            loss = IoU_Loss(true_boxes, pred)

        # Backpropagation.
        grads = tape_box.gradient(loss, self.nn_box.trainable_variables)
        self.box_optimizer.apply_gradients(zip(grads, self.nn_box.trainable_variables))
        return loss
    
    def create_model(self, frames):
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
        x = Dropout(0.2)(x)
        x = Dense(256, activation = 'relu')(x)
        x = Dense(4 * frames)(x)
        return Model(inputs, x, name=MODEL_NAME)


    def fit(self):
        pass


    def testing(self, dataset):
        """ Проверка работы """
        # todo не проверял
        for ii, cc in dataset.take(1):
            #обрабатывем целый батч, используем только пять элементов
            pred = model.nn_box(ii)
            plt.figure(figsize=(10, 6))
            
            for num in range(3):
                i = ii[num]
                
                # pred = tf.reshape(pred, [-1, self.frames, 3])
                pred = tf.reshape(pred, [-1, self.frames, 4])
                c = pred[num]

                ax = plt.subplot(1, 5, num+1)
                #переход в numpy для работы в opencv
                i = i.numpy()
                c = c.numpy()
                c = (c+1)/2*128 #обратно из от -1...1 к 0...64
                c = c.astype(np.int16)  #для opencv
                for bb in c:
                    bb0 = min(bb[0] ,bb[2])
                    bb2 = max(bb[0] ,bb[2])
                    i = cv2.rectangle(i ,(bb0 ,bb[1] ),(bb2, bb[1] + (bb2- bb0)),(0,1,0),1)
                plt.imshow(i)
            plt.show()



def main(use_checkpoint=1):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model_chkpt_step_path = os.path.join(DEFAULT_MODEL_DIR, f"epoch{{epoch:02d}}-{MODEL_NAME}.h5")
    model_save_path = os.path.join(DEFAULT_MODEL_DIR, f"{MODEL_NAME}.h5")

    # Создание датасета
    train_dataset, test_dataset = get_dataset()

    # examples_train = sum(1 for _ in train_dataset)
    # examples_test = sum(1 for _ in train_dataset)
    # print(f"Набор данных train: {examples_train} | test: {examples_test}")

    # todo сначала сделать без нормализации, а потом с ней
    
    model = LocalizationModel()

    # todo не проверял
    # пример работы
    
    print(train_dataset)
    for i, c in train_dataset:
        print("пример работы")
        print(tf.reduce_mean(model.training_step(i, c)))
        print("окончание")

    # todo не проверял
    model.testing(train_dataset)

    # todo не проверял
    # обучение, запихать его в класс модели
    # попробовать просто модель сделать с вот этой функцией потерь
    # from IPython.display import clear_output
    # hist = np.array(np.empty([0]))
    # epochs = 100
    # ff = 0
    # for epoch in range(1, epochs + 1):
    #     loss = 0
    #     lc = 0
    #     for step, (i, c) in enumerate(train_dataset):
    #         loss+=tf.reduce_mean(model.training_step(i,c))
    #         lc+=1
    #     clear_output(wait=True)
    #     print(epoch)
    #     hist = np.append(hist, loss/lc)
    
    #     plt.plot(np.arange(0,len(hist)), hist)
    #     plt.show()
    #     model.testing()

    # todo не проверял
    # model.nn_box.save('my_bb_model.h5')


    # unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])

    # if use_checkpoint == 1 and os.path.exists(model_save_path):
    #     print(f"\nВключено использование чекпоинтов.\nЗагрузка модели...")
    #     model = load_model(model_save_path, custom_objects=custom_objects)
    #     print(f"Модель загружена из файла: {model_save_path}\n")


    # checkpoint_step = ModelCheckpoint(model_chkpt_step_path, save_freq='epoch')
    ## Обучаем нейронную сеть и сохраняем результат
    # unet_like.fit(train_dataset, validation_data=test_dataset, epochs=25,
    #                              callbacks=[checkpoint_step])
    # unet_like.save(model_save_path)


if __name__ == '__main__':
    main()
