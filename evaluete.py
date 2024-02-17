import os

from train import dice_bce_mc_loss, dice_mc_loss, dice_mc_metric
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import glob
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, './saved-model/hangul-localizator.h5')
DEFAULT_IMAGE_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-images/')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-output/')

CLASSES = 1
CHANNELS = 1
SAMPLE_SIZE = (128, 128)

OUTPUT_SIZE = (128, 128)

## Проверим работу сети на всех кадрах из видео
rgb_colors = [
    (0,   0,   0)
]

custom_objects={'dice_bce_mc_loss': dice_bce_mc_loss,
                'dice_mc_metric': dice_mc_metric,
                'dice_mc_loss': dice_mc_loss}


if __name__ == '__main__':
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(DEFAULT_IMAGE_DIR, '*.jpeg')))
    print(f'Found {len(frames)} frames')

    unet_like = load_model(DEFAULT_MODEL_DIR, custom_objects=custom_objects)

    for filename in frames:
        frame = imread(filename)
        sample = resize(frame, SAMPLE_SIZE)
        
        predict = unet_like.predict(sample.reshape((1,) +  SAMPLE_SIZE + (CHANNELS,)))
        predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))


        predict = predict.squeeze(-1)  # Убрать последнюю размерность (каналы)
        predict = (predict * 255).astype(np.uint8)
        image = Image.fromarray(predict)
        image = ImageOps.invert(image)
        image.save(f'{DEFAULT_OUTPUT_DIR}/{os.path.basename(filename)}')



