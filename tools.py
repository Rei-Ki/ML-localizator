import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, \
    Dropout, ReLU, Conv2DTranspose


def create_model(SAMPLE_SIZE, CLASSES, channels=3):
    ## Обозначим основные блоки модели
    def input_layer():
        return tf.keras.layers.Input(shape=SAMPLE_SIZE + (channels,))

    def downsample_block(filters, size, batch_norm=True):
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()
        
        result.add(
            Conv2D(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False))

        if batch_norm:
            result.add(BatchNormalization())
        
        result.add(LeakyReLU())
        return result

    def upsample_block(filters, size, dropout=False):
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()
        
        result.add(
            Conv2DTranspose(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False))

        result.add(BatchNormalization())
        
        if dropout:
            result.add(Dropout(0.25))
        
        result.add(ReLU())
        return result

    def output_layer(size):
        initializer = tf.keras.initializers.GlorotNormal()
        return Conv2DTranspose(CLASSES, size, strides=2, padding='same',
            kernel_initializer=initializer, activation='sigmoid')


    ## Построим U-NET подобную архитектуру
    inp_layer = input_layer()

    downsample_stack = [
        downsample_block(64, 4, batch_norm=False),
        downsample_block(128, 4),
        downsample_block(256, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
    ]

    upsample_stack = [
        upsample_block(512, 4, dropout=True),
        upsample_block(512, 4, dropout=True),
        upsample_block(512, 4, dropout=True),
        upsample_block(256, 4),
        upsample_block(128, 4),
        upsample_block(64, 4)
    ]

    out_layer = output_layer(4)

    # Реализуем skip connections
    x = inp_layer

    downsample_skips = []

    for block in downsample_stack:
        x = block(x)
        downsample_skips.append(x)
        
    downsample_skips = reversed(downsample_skips[:-1])

    for up_block, down_block in zip(upsample_stack, downsample_skips):
        x = up_block(x)
        x = tf.keras.layers.Concatenate()([x, down_block])

    out_layer = out_layer(x)

    unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)
    return unet_like


def show_images_and_masks(dataset, classes, colors):
    ## Посмотрим на содержимое набора данных
    images_and_masks = list(dataset.take(5))

    fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(16, 6))

    for i, (image, masks) in enumerate(images_and_masks):
        ax[0, i].set_title('Image')
        ax[0, i].set_axis_off()
        ax[0, i].imshow(image)
            
        ax[1, i].set_title('Mask')
        ax[1, i].set_axis_off()    
        ax[1, i].imshow(image/1.5)
    
        for channel in range(classes):
            contours = measure.find_contours(np.array(masks[:,:,channel]))
            for contour in contours:
                ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[channel])

    plt.show()
    plt.close()

