import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import time
# import json
import ijson
import numpy as np
import pickle
import glob
import tensorflow as tf
import argparse
import io
import random
import tensorflow as tf
from multiprocessing import Pool, current_process

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, './image-data/labels-map.csv')
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_PATH, './image-data/')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, './output-tfrecords/')

DEFAULT_NUM_SHARDS_TRAIN = 6
DEFAULT_NUM_SHARDS_TEST = 2
DEFAULT_WORKERS = 8
DEFAULT_IS_PROCESS = 0
IMAGE_SIZE = (128, 128)

FRAMES = 10


def load_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)/255
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def coords_to_frames(array):
    if array.shape[0] >= FRAMES:
        raise ValueError(f"Количество символов на изображении превышает количество рамок {FRAMES}")
    if array.shape[0] <= FRAMES:
        need = FRAMES - array.shape[0]
        if need == 0:
            return array
        
        for i in range(need):
            index = np.random.choice(array.shape[0])
            random_row = array[index, :]
            array = np.concatenate((array, [random_row]), axis=0)
    return array


class TFRecordsConverter(object):
    """Class that handles converting images to TFRecords."""
    
    def __init__(self, images_dir, output_dir, num_shards_train, num_shards_test, workers):

        self.output_dir = output_dir
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test
        self.workers = workers

        os.makedirs(self.output_dir, exist_ok=True)

        print("Загрузка данных...")
        # Получить списки изображений и меток.
        self.filenames, self.coords = self.process_image_labels(images_dir)        

        print("Загрузка имен изображений...")
        self.filenames  = pickle.load(open('filenames.pkl', 'rb'))
        print("Загрузка координат...")
        self.coords = pickle.load(open('coords.pkl', 'rb'))
        print("Данные загружены!")
        
        if len(self.filenames) != len(self.coords):
            print("Массивы меток и свойств не равны")
            exit()
        
        # Счетчик для общего количества обработанных изображений.
        self.counter = 0
        self.all_images = len(self.filenames)


    def process_image_labels(self, images_dir):
        """Эта функция создаст два перемешанных списка для изображений и меток.

        Индекс каждого изображения в списке изображений будет иметь соответствующую
        метку на том же индексе в списке меток.
        """
        print(f"Подсчет количества изображений")
        csv_path = os.path.join(images_dir, './labels-map.csv')
        labels_csv = io.open(csv_path, 'r', encoding='utf-8')

        images_path = []
        coords = []
        print(f"Уже обработано: {len(images_path)}")
        # todo посмотреть что бы в координатах был массив из 4 элементов на одной позиции
        
        last_save = 0
        for i, file_name in enumerate(labels_csv):
            # if i <= len(images_path):
            #     continue
            
            file_name, json_name = file_name.strip().split(',')
            
            with open(json_name, 'r') as f:  # Загружаем JSON файл
                objects = ijson.items(f, 'shapes.item.points')
                coords_local = np.array(list(objects)).astype(np.float32)
                
            coords_local = coords_to_frames(coords_local)
            
            coords.append(coords_local)
            images_path.append(file_name)

            if i != 0 and i % 100000 == 0:
                pickle.dump(images_path, open('filenames.pkl', 'wb'))
                pickle.dump(coords, open('coords.pkl', 'wb'))
                last_save = i
            print(f"Пройдено {i:_} изображений | {last_save = :_}", end='\r')

        pickle.dump(images_path, open('filenames.pkl', 'wb'))
        pickle.dump(coords, open('coords.pkl', 'wb'))
        return images_path, coords


    def write_tfrecords_file(self, output_path, indices):
        """Записывает файл TFRecords."""
        process_name = current_process().name
        print(f'Процесс {process_name} начал запись в файл {output_path}')

        work_len_indices = len(indices)
        inner_counter = 0
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in indices:
                filename = self.filenames[i]
                
                image = load_image(filename)
                coords = self.coords[i]

                serialized_img = tf.io.serialize_tensor(image).numpy()
                serialized_cords = tf.io.serialize_tensor(coords).numpy()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
                    'coords': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_cords]))
                }))
                writer.write(example.SerializeToString())
                
                inner_counter += 1
                if not inner_counter % 1000:
                    percent = inner_counter/work_len_indices*100
                    print(f'Обработано {inner_counter:_} / {work_len_indices:_} ({percent:.2f}) изображений ({process_name})...')
        print(f'Завершена обработка {inner_counter} изображений ({process_name})')


    def create_file(self, args):
        try:
            start, stop, name = args
            print(f'Запуск процесса для создания {name}')

            shard_path = os.path.join(self.output_dir, f'{name}.tfrecords')
            file_indices = tf.range(start, stop, dtype=tf.int32)
            self.write_tfrecords_file(shard_path, file_indices)
        except Exception as e:
            print(f"Ошибка в процессе {name}: {e}")



    def convert(self):
        num_files_total = len(self.filenames)
        # Allocate about 15 percent of images to testing
        num_files_test = int(num_files_total * .15)
        # About 85 percent will be for training.
        num_files_train = num_files_total - num_files_test

        # Трейновая выборка
        files_per_shard = int(tf.math.ceil(num_files_train / self.num_shards_train))
        start = 0
        indexes = []
        for i in range(1, self.num_shards_train):    
            indexes.append((start, start+files_per_shard))
            start = start + files_per_shard
        indexes.append((start, num_files_train))

        args_list = [(indexes[shard][0], indexes[shard][1], f"train-{shard}") 
                     for shard in range(0, self.num_shards_train)]
        
        # Тестовая выборка
        files_per_shard = tf.math.ceil(num_files_test / self.num_shards_test)
        start = num_files_train
        indexes = []
        for i in range(1, self.num_shards_test):    
            indexes.append((start, start+files_per_shard))
            start = start + files_per_shard
        indexes.append((start, num_files_total))

        test_pool = [(indexes[shard][0], indexes[shard][1], f"test-{shard}") 
                     for shard in range(0, self.num_shards_test)]
        
        for ap in test_pool:
            args_list.append(ap)

        with Pool(processes=self.workers) as pool:
            pool.map(self.create_file, args_list)


        print(f'\nProcessed {self.counter} total images...')
        print(f'Number of training examples: {num_files_train}')
        print(f'Number of testing examples: {num_files_test}')
        print(f'TFRecords files saved to {self.output_dir}\n')


    def non_process_convert(self):
        """This function will drive the conversion to TFRecords.

        Here, we partition the data into a training and testing set, then
        divide each data set into the specified number of TFRecords shards.
        """

        num_files_total = len(self.filenames)

        # Allocate about 15 percent of images to testing
        num_files_test = int(num_files_total * .15)

        # About 85 percent will be for training.
        num_files_train = num_files_total - num_files_test

        print('\nProcessing training set TFRecords...')

        files_per_shard = int(tf.math.ceil(num_files_train / self.num_shards_train))
        start = 0
        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir, f'train-{i}.tfrecords')
            # Get a subset of indices to get only a subset of images/labels for
            # the current shard file.
            file_indices = tf.range(start, start+files_per_shard, dtype=tf.int32)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = tf.range(start, num_files_train, dtype=tf.int32)
        final_shard_path = os.path.join(self.output_dir, f'train-{self.num_shards_train}.tfrecords')
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('\nProcessing testing set TFRecords...')

        files_per_shard = tf.math.ceil(num_files_test / self.num_shards_test)
        start = num_files_train
        for i in range(1, self.num_shards_test):
            shard_path = os.path.join(self.output_dir, f'test-{i}.tfrecords')
            file_indices = tf.range(start, start+files_per_shard, dtype=tf.int32)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = tf.range(start, num_files_total, dtype=tf.int32)
        final_shard_path = os.path.join(self.output_dir, f'test-{self.num_shards_test}.tfrecords')
        self.write_tfrecords_file(final_shard_path, file_indices)

        print(f'Number of training examples: {num_files_train}')
        print(f'Number of testing examples: {num_files_test}')
        print(f'TFRecords files saved to {self.output_dir}\n')


def main(images_dir, output_dir, num_shards_train, num_shards_test, workers, no_parallel):
    """Generate Hangul images."""

    converter = TFRecordsConverter(images_dir, output_dir, num_shards_train, num_shards_test, workers)
    if no_parallel == 0:
        converter.convert()
    else:
        converter.non_process_convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, dest='images_dir',
                        default=DEFAULT_IMAGES_DIR,
                        help='Input directory which stored images files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecords files.')
    parser.add_argument('--num-shards-train', type=int,
                        dest='num_shards_train',
                        default=DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into.')
    parser.add_argument('--num-shards-test', type=int,
                        dest='num_shards_test',
                        default=DEFAULT_NUM_SHARDS_TEST,
                        help='Number of shards to divide testing set '
                             'TFRecords into.')
    parser.add_argument('--workers', type=int, dest='workers',
                    default=DEFAULT_WORKERS,
                    help='Num of threads to use for generating images.')
    parser.add_argument('--no-parallel', type=int, dest='no_parallel',
                    default=DEFAULT_IS_PROCESS,
                    help='Use non-parallel processing.')
    args = parser.parse_args()
    main(args.images_dir, args.output_dir, args.num_shards_train, args.num_shards_test, args.workers, args.no_parallel)
