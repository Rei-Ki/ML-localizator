import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from tools import create_model, show_images_and_masks
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, \
    Dropout, ReLU, Conv2DTranspose
import matplotlib.pyplot as plt

from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Стандартные пути
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangul.txt')
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model')

DEFAULT_LABEL_CSV_FILE = os.path.join(SCRIPT_PATH, './image-data/labals-map.csv')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, './image-data')
DEFAULT_MASK_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-masks')

MODEL_NAME = 'hangul-localizator'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

TRAIN_EXAMPLES = 2000

# Подготовим набор данных для обучения
# Это будет определяться количеством записей в данном файле меток
# CLASSES = 8
# labels = io.open(DEFAULT_LABEL_FILE, 'r', encoding='utf-8').read().splitlines()
# CLASSES = len(labels)
CLASSES = 1

# COLORS = ['black', 'red', 'lime', 'blue', 'orange', 'pink', 'cyan', 'magenta']
# SAMPLE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
COLORS = ['pink']
# SAMPLE_SIZE = (256, 256)
SAMPLE_SIZE = (128, 128)

# OUTPUT_SIZE = (1080, 1920)
OUTPUT_SIZE = (128, 128)


@tf.function
def load_images(image, mask):
    # print(f'Loading images: {image}\nLoading masks: {mask}' )
    # tf.print(image)
    # сделать как разметка на видосе, то есть каждый класс цифрами помечен
    # может это и не надо, один же объект границы слова
    # тогда больше данных надо

    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_jpeg(mask)
    # mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    masks = []
    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))
    
    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

    return image, masks

@tf.function
def augmentate_images(image, masks):   
    # random_crop = tf.random.uniform((), 0.3, 1)
    # image = tf.image.central_crop(image, random_crop)
    # masks = tf.image.central_crop(masks, random_crop)
    
    # random_flip = tf.random.uniform((), 0, 1)    
    # if random_flip >= 0.5:
    #     image = tf.image.flip_left_right(image)
    #     masks = tf.image.flip_left_right(masks)
    
    image = tf.image.resize(image, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)
    
    return image, masks

## Определим метрики и функции потерь
def dice_mc_metric(a, b):
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)
    
    dice_summ = 0
    
    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_summ += numenator / denomerator
        
    avg_dice = dice_summ / CLASSES
    
    return avg_dice

def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)

def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)


custom_objects={'dice_bce_mc_loss': dice_bce_mc_loss,
                'dice_mc_metric': dice_mc_metric,
                'dice_mc_loss': dice_mc_loss}


def main(use_checkpoint=1):
    model_chkpt_step_path = os.path.join(DEFAULT_MODEL_DIR, f"epoch{{epoch:02d}}-{MODEL_NAME}.h5")
    model_save_path = os.path.join(DEFAULT_MODEL_DIR, f"{MODEL_NAME}.h5")

    # images = sorted(glob.glob(os.path.join(SCRIPT_PATH, './dataset/images/*.jpg')))
    # masks = sorted(glob.glob(os.path.join(SCRIPT_PATH, './dataset/masks/*.png')))
    images = sorted(glob.glob(os.path.join(SCRIPT_PATH, './image-data/hangul-images/*.jpeg')))
    masks = sorted(glob.glob(os.path.join(SCRIPT_PATH, './image-data/hangul-masks/*.jpeg')))
    
    examples = len(images)

    images_dataset = tf.data.Dataset.from_tensor_slices(images)
    masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

    dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

    dataset = dataset \
        .map(load_images) \
        .map(augmentate_images) \
        .repeat(60)


    num_train_examples = int(examples * 0.8)
    print(f"Набор данных: {examples}")
    print(f"Набор данных для обучения: {num_train_examples}")
    print(f"Набор данных для теста: {examples - num_train_examples}")

    # Разделим набор данных на обучающий и проверочный
    train_dataset = dataset.take(num_train_examples).cache()
    test_dataset = dataset.skip(num_train_examples) \
        .take(examples - num_train_examples).cache()
    
    train_dataset = train_dataset.batch(8)
    test_dataset = test_dataset.batch(8)

    # show_images_and_masks(dataset, CLASSES, COLORS)

    unet_like = create_model(SAMPLE_SIZE=SAMPLE_SIZE, CLASSES=CLASSES, channels=1)


    unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])

    checkpoint_step = ModelCheckpoint(model_chkpt_step_path, save_freq='epoch')

    if use_checkpoint == 1 and os.path.exists(model_save_path):
        print(f"\nВключено использование чекпоинтов.\nЗагрузка модели...")
        model = load_model(model_save_path, custom_objects=custom_objects)
        print(f"Модель загружена из файла: {model_save_path}\n")


    ## Обучаем нейронную сеть и сохраняем результат
    unet_like.fit(train_dataset, validation_data=test_dataset, epochs=25,
                                 callbacks=[checkpoint_step])

    unet_like.save(model_save_path)


if __name__ == '__main__':
    main()
