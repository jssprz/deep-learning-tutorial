import os
import pickle
import time

import numpy as np
from scipy.misc import imsave


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def save_image(data, label, filename, folder, log):
    # reshape image channels information, data[0] is R, [1] is G, [2] is B
    data = data.reshape((3, 32, 32))

    # change format for scipy
    img = np.zeros((32, 32, 3))
    for i in range(3):
        img[:, :, i] = data[i]

    # save image image
    img_path = os.path.join(os.path.abspath(folder), f'{label}_{filename}')
    imsave(img_path, img)

    log.write(f'{img_path}\t{label}\n')


def convert_batchs_to_imgs(batch_files, to_save_list_path, to_save_images_dir):
    # create folder if not exist
    if not os.path.isdir(to_save_images_dir):
        os.mkdir(to_save_images_dir)

    log = open(to_save_list_path, 'w')

    # process batch
    for batch_file in batch_files:
        batch = unpickle(batch_file)
        data = batch[b'data']
        labels = batch[b'labels']
        filenames = batch[b'filenames']

        print(f'Converting {batch_file}')
        t = time.time()

        for i in range(len(data)):
            save_image(data[i], labels[i], filenames[i].decode("utf-8"), to_save_images_dir, log)

            if i % 1000 == 999:
                print(f'{i + 1} imgs converted')

        print(f'done converting {batch_file} ({int(time.time() - t)} seconds)')

    log.close()


def main():
    data_dir = '../../../../datasets/CIFAR-10/cifar-10-batches-py'
    train_files = [os.path.join(data_dir, f'data_batch_{i+1}') for i in range(5)]
    test_files = [os.path.join(data_dir, 'test_batch')]

    print('\nConverting train images')
    convert_batchs_to_imgs(train_files, os.path.join(data_dir, 'train.txt'), os.path.join(data_dir, 'train_images'))
    print('\nConverting test images')
    convert_batchs_to_imgs(test_files, os.path.join(data_dir, 'test.txt'), os.path.join(data_dir, 'test_images'))


if __name__ == '__main__':
    main()
