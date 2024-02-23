import os
import glob
from pathlib import Path
import random
import argparse
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from multiprocessing import Process, current_process
import matplotlib.pyplot as plt
import cv2
import time

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, './fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, './image-data')
DEFAULT_MASK_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-masks')
DEFAULT_WORKERS = 1

# Width and height of the resulting image.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

NUM_SYMBOLS = 10


default_json = {
  "version": "4.5.7",
  "flags": {},
  "shapes": [],
  "imagePath": "",
  "imageData": None,
  "imageHeight": IMAGE_HEIGHT,
  "imageWidth": IMAGE_WIDTH
}

default_points = {
      "label": "1",
      "points": [],
      "group_id": None,
      "shape_type": "rectangle",
      "flags": {}
    }


def get_actual_font(w, h, font_i, font_size, characters):
    while w > IMAGE_WIDTH - 20 or h > IMAGE_HEIGHT:
        font_size -= 5
        font = ImageFont.truetype(font_i, font_size)
        w, h = font.getbbox("".join(characters))[2:4]
    return font_size


def generate_images_on_font(total_count, font_i, characters, fonts_sizes, name, image_dir, labels_csv, thread_work):
    for font_size in fonts_sizes:
        total_count += 1
        image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
        drawing = ImageDraw.Draw(image)

        # todo потом сделать еще и разное расстояние между буквами
        spacing = 2
        points = []

        # Получаем размеры всего слова
        font = ImageFont.truetype(font_i, font_size)
        w, h = font.getbbox("".join(characters))[2:4]
        
        # Проверяем, влезает ли слово в изображение
        font_size = get_actual_font(w, h, font_i, font_size, characters)
        
        for i, character in enumerate(characters):
            font = ImageFont.truetype(font_i, font_size)
            w, h = font.getbbox(character)[2:4]
            # Расположите символы по горизонтали на изображении
            x = ((i + 1) * (IMAGE_WIDTH - spacing) // (len(characters) + 1)) - w // 2 + spacing // 2
            y = (IMAGE_HEIGHT - h) / 2
            drawing.text((x, y), character, fill=(0), font=font)

            # Запись точек прямоугольника
            point = default_points.copy()
            point["label"] = character
            point["points"] = [x, y, x + w, y + h]
            points.append(point)
        
        file_string = f'hangul_{name}_{total_count}.jpeg'
        file_json = f'hangul_{name}_{total_count}.json'
        file_path = os.path.join(image_dir, file_string)
        file_json = os.path.join(image_dir, file_json)
        image.save(file_path, 'JPEG')
        
        # Преобразование кортежа в список
        dumped = default_json.copy()
        dumped["shapes"] = points
        dumped["imagePath"] = file_path
        # Сохранение списка в JSON файл
        with open(file_json, 'w') as f:
            json.dump(dumped, f)
        
        labels_csv.write(f'{file_path},{file_json}\n')

        if DEFAULT_WORKERS == 1:
            print(f'{total_count} / {thread_work} ({(total_count/thread_work*100):.2f}%) ({name})', end='\r')
    return total_count


def generate_hangul_images(output_dir, fonts, image_dir, list_labels, index_start, name):
    all_fonts_sizes = [53, 50, 46, 36, 30]
    # all_fonts_sizes = [46]
    # thread_work = NUM_SYMBOLS * len(list_labels) * len(fonts) * len(all_fonts_sizes)
    thread_work = len(list_labels) * len(fonts) * len(all_fonts_sizes)
    
    print(f'{0} / {thread_work} ({(0/thread_work*100):.2f}%) ({name})', end='\r')
    with open(os.path.join(output_dir, f'thread-{name}-labels-map.csv'), 'w', encoding='utf-8') as labels_csv:
        total_count = index_start
        prev_count = index_start

        for _ in list_labels:
            characters = random.sample(list_labels, random.randint(1, NUM_SYMBOLS + 1))
            
            if total_count - prev_count > 5000:
                prev_count = total_count
                if DEFAULT_WORKERS == 1:
                    print(f'{total_count} / {thread_work} ({(total_count/thread_work*100):.2f}%) ({name})', end='\r')
                else:
                    print(f'{total_count} / {thread_work} ({(total_count/thread_work*100):.2f}%) ({name})')
            
            for font_i in fonts:
                total_count = generate_images_on_font(total_count, font_i, 
                        characters, all_fonts_sizes, name, image_dir, labels_csv, thread_work)


def concat_outputs(output_dir):
    data_dir = Path(output_dir)

    df = pd.concat([pd.read_csv(f, header=None) 
                    for f in data_dir.glob("thread-*.csv")], ignore_index=True)

    print(df)
    df.to_csv(os.path.join(output_dir, f'labels-map.csv'), header=None, index=False)


def main(label_file, fonts_dir, output_dir, workers):
    """Generate Hangul images."""
    DEFAULT_WORKERS = workers

    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'hangul-images')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(DEFAULT_MASK_DIR, exist_ok=True)

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))


    middle = len(labels) // workers

    list_works = []
    list_indexes = []
    for i in range(0, len(labels), middle):
        if i + middle + 200 > len(labels):
            list_works.append(labels[i:])
            list_indexes.append(i)
            break
        list_works.append(labels[i:i + middle])
        list_indexes.append(i)


    process = []
    for i in range(workers):
        print(f'Запуск процесса {i}')
        process.append(
            Process(target=generate_hangul_images, name=f'process {i}', daemon=True,
                    args=(output_dir, fonts, image_dir, list_works[i], list_indexes[i], f'ps{i}',)))

    for i in range(workers):
        process[i].start()

    for i in range(workers):
        process[i].join()

    print('\nВсе процессы завершены.\nОбъединение результатов...')
    concat_outputs(output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and label CSV file.')
    parser.add_argument('--workers', type=int, dest='workers',
                        default=DEFAULT_WORKERS,
                        help='Num of threads to use for generating images.')
    args = parser.parse_args()
    main(args.label_file, args.fonts_dir, args.output_dir, args.workers)
