import os
import glob
from pathlib import Path
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from multiprocessing import Process, current_process

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, './fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, './image-data')
DEFAULT_MASK_DIR = os.path.join(SCRIPT_PATH, './image-data/hangul-masks')
DEFAULT_WORKERS = 1

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 12

# Width and height of the resulting image.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

NUM_IMAGES = 1

def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = np.random.default_rng()
    shape = image.shape

    dx = gaussian_filter(
        (random_state.random(shape) * 2 - 1),
        sigma, mode="constant", cval=0
    ) * alpha
    dy = gaussian_filter(
        (random_state.random(shape) * 2 - 1),
        sigma, mode="constant", cval=0
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

def generate_hangul_images_for_mask(output_dir, fonts, image_dir, list_labels, index_start, name):
    import random

    all_fonts_sizes = [50]

    with open(os.path.join(output_dir, f'thread-{name}-labels-map.csv'), 'w', encoding='utf-8') as labels_csv:
        total_count = index_start
        prev_count = index_start
        
        font_i = fonts[0]
        for _ in range(NUM_IMAGES):  # NUM_IMAGES - это количество изображений, которые вы хотите создать
            # Выберите 5 случайных символов из list_labels
            
            characters = random.sample(list_labels, random.randint(2, 5))
            # Print image count roughly every 5000 images.
            if total_count - prev_count > 5000:
                prev_count = total_count
                print(f'{total_count} изображений сгенерировано ({name})...')
            
            for font_size in all_fonts_sizes:
                total_count += 1
                image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
                drawing = ImageDraw.Draw(image)
                
                for i, character in enumerate(characters):
                    font = ImageFont.truetype(font_i, font_size)
                    w, h = font.getbbox(character)[2:4]
                    # Расположите символы по горизонтали на изображении
                    drawing.text(
                        (i * IMAGE_WIDTH // 5 + (IMAGE_WIDTH // 5 - w) / 2, (IMAGE_HEIGHT - h) / 2),
                        character, fill=(0), font=font
                    )
                    
                file_string = f'hangul_{name}_{total_count}.jpeg'
                file_path = os.path.join(image_dir, file_string)
                image.save(file_path, 'JPEG')
                # Сохраните каждую маску в отдельном файле

                # Запишите все символы в CSV
                labels_csv.write(f'{file_path},{"".join(characters)}\n')


def generate_hangul_images_mask_many(output_dir, fonts, image_dir, list_labels, index_start, name):
    import random

    all_fonts_sizes = [53]

    with open(os.path.join(output_dir, f'thread-{name}-labels-map.csv'), 'w', encoding='utf-8') as labels_csv:
        total_count = index_start
        prev_count = index_start

        for _ in range(NUM_IMAGES):  # NUM_IMAGES - это количество изображений, которые вы хотите создать
            # Выберите 5 случайных символов из list_labels
            # characters = random.sample(list_labels, random.randint(2, 5))
            characters = random.sample(list_labels, 1)
            # Print image count roughly every 5000 images.
            if total_count - prev_count > 5000:
                prev_count = total_count
                print(f'{total_count} изображений сгенерировано ({name})...')
            
            for font_i in fonts:
                for font_size in all_fonts_sizes:
                    total_count += 1
                    image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
                    drawing = ImageDraw.Draw(image)
                    masks = []  # Список для хранения масок каждого символа
                    
                    for i, character in enumerate(characters):
                        font = ImageFont.truetype(font_i, font_size)
                        w, h = font.getbbox(character)[2:4]
                        # Расположите символы по горизонтали на изображении
                        drawing.text(
                            # (i * IMAGE_WIDTH // 5 + (IMAGE_WIDTH // 5 - w) / 2, (IMAGE_HEIGHT - h) / 2),
                            ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                            character, fill=(0), font=font
                        )
                        # Создайте маску для этого символа

                        mask = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
                        mask_drawing = ImageDraw.Draw(mask)
                        # x1 = i * IMAGE_WIDTH // 5 + (IMAGE_WIDTH // 5 - w) / 2
                        x1 = (IMAGE_WIDTH-w)/2
                        y1 = (IMAGE_HEIGHT - h) / 2
                        x2 = x1 + w
                        y2 = y1 + h
                        # Рисуйте прямоугольник вокруг символа
                        mask_drawing.rectangle([x1, y1, x2, y2], outline=(0))

                        masks.append(mask)
                    # file_string = f'hangul_{name}_{total_count}.jpeg'
                    # file_path = os.path.join(image_dir, file_string)
                    # image.save(file_path, 'JPEG')
                    # Сохраните каждую маску в отдельном файле

                    masks_name = []
                    for j, mask in enumerate(masks):
                        mask_file_string = f'hangul_{name}_{total_count}_mask{j}.jpeg'
                        masks_name.append(os.path.join(DEFAULT_MASK_DIR, mask_file_string))
                        mask_file_path = os.path.join(DEFAULT_MASK_DIR, mask_file_string)  # Предполагается, что mask_dir - это путь к каталогу для сохранения масок
                        
                        file_string = f'hangul_{name}_{total_count}_image{j}.jpeg'
                        file_path = os.path.join(image_dir, file_string)
                        image.save(file_path, 'JPEG')
                        
                        mask.save(mask_file_path, 'JPEG')
                        labels_csv.write(f'{file_path},{mask_file_path}\n')

                    # Запишите все символы в CSV
                    # labels_csv.write(f'{file_path},{",".join(masks_name)}\n')


def generate_hangul_images(output_dir, fonts, image_dir, list_labels, index_start, name):
    print(f'\nStart generating images...')
    
    # all_fonts_sizes = [53, 50, 46, 36, 30]
    all_fonts_sizes = [53]

    with open(os.path.join(output_dir, f'thread-{name}-labels-map.csv'), 'w', encoding='utf-8') as labels_csv:
        total_count = index_start
        prev_count = index_start
        for character in list_labels:
            # Print image count roughly every 5000 images.
            if total_count - prev_count > 5000:
                prev_count = total_count
                print(f'{total_count} изображений сгенерировано ({name})...')

            for font_i in fonts:
                for font_size in all_fonts_sizes:
                    total_count += 1
                    image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
                    font = ImageFont.truetype(font_i, font_size)
                    drawing = ImageDraw.Draw(image)
                    w, h = font.getbbox(character)[2:4]
                    drawing.text(
                        ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                        character, fill=(255), font=font
                    )
                    file_string = f'hangul_{name}_{total_count}.jpeg'
                    file_path = os.path.join(image_dir, file_string)
                    image.save(file_path, 'JPEG')
                    labels_csv.write(f'{file_path},{character}\n')

                    # for _ in range(DISTORTION_COUNT):
                    #     total_count += 1
                    #     file_string = f'hangul_{name}_{total_count}.jpeg'
                    #     file_path = os.path.join(image_dir, file_string)
                    #     arr = np.array(image)
                        
                    #     distorted_array = elastic_distort(
                    #         arr, alpha=random.randint(25, 36),
                    #         sigma=random.randint(5, 6)
                    #     )
                    #     distorted_image = Image.fromarray(distorted_array)

                    #     # Поворот изображения
                    #     rotated_image = distorted_image.rotate(random.uniform(-15, 15))

                    #     # Смещение изображения
                    #     dx, dy = random.randint(-10, 12), random.randint(-8, 15)
                    #     matrix = (1, 0, dx, 0, 1, dy)
                    #     transformed_image = rotated_image.transform(rotated_image.size, Image.AFFINE, matrix)

                    #     transformed_image.save(file_path, 'JPEG')
                    #     labels_csv.write(f'{file_path},{character}\n')

        print(f'Процесса завершил генерацию {total_count} изображений.\n')


def concat_outputs(output_dir):
    data_dir = Path(output_dir)

    df = pd.concat([pd.read_csv(f, header=None) 
                    for f in data_dir.glob("thread-*.csv")], ignore_index=True)

    print(df)
    df.to_csv(os.path.join(output_dir, f'labels-map.csv'), header=None, index=False)


def main(label_file, fonts_dir, output_dir, workers):
    """Generate Hangul images."""

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
            Process(target=generate_hangul_images_mask_many, name=f'process {i}', daemon=True,
                    args=(output_dir, fonts, image_dir, list_works[i], list_indexes[i], f'ps{i}',)))
            # Process(target=generate_hangul_images, name=f'process {i}', daemon=True,
            #         args=(output_dir, fonts, image_dir, list_works[i], list_indexes[i], f'ps{i}',)))

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
