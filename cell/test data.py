# _*_ conding:utf-8 _*_
from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = '/home/momoh/mabocombinedimgs22/'

image_rows = 420
image_cols = 580

def create_test_data2():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)  #文件名列表
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)  #np.ndarray中参数表示的是维度，默认值为零
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        img = imread(os.path.join(train_data_path, image_name))	#(width,height,channel)
        img_mask = imread(os.path.join(train_data_path, image_mask_name))

        img =img[:,:,1]			#(width,height)
        img_mask=img_mask[:,:,1]
        img = np.array([img])		#(1,width,height)
        img_mask = np.array([img_mask])

        imgs[i] = img	#(i,1,width,height)
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print(total)

    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    print('Saving to .npy files done.')

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        img = imread(os.path.join(train_data_path, image_name))
        img_mask = imread(os.path.join(train_data_path, image_mask_name))

        img =img[:,:,1]
        img_mask=img_mask[:,:,1]
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print(total)

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)/2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        img_id = int(image_name.split('.')[0])#image_name
        img = imread(os.path.join(train_data_path, image_name))
        img =img[:,:,1]

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id,imgs_mask_test

if __name__ == '__main__':
    #create_train_data()
    create_test_data()
    create_test_data2()



