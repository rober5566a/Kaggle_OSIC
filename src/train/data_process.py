from utils import *
import os
import cv2
import pickle
import gdcm
import pydicom
import numpy as np
from Model.find_file_name import get_filenames


def statistic(csv_file):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    for tag, idx in [('WEEK', 1), ('FVC', 2), ('PERCENT', 3), ('AGE', 4)]:
        data = [float(e[idx]) for e in content]
        print('{} = ({}, {})'.format(tag, np.mean(data), np.std(data)))


def onehot(idx, length):
    temp = np.zeros((length), np.float32)
    temp[idx] = 1
    return temp


def normalize_info(line):
    temp = np.concatenate((
        np.array([codec_w.encode(line[1]), codec_f.encode(line[2]),
                  codec_p.encode(line[3]), codec_a.encode(line[4])], np.float32),
        onehot(['Male', 'Female'].index(line[5]), 2),
        onehot(['Never smoked', 'Ex-smoker', 'Currently smokes'].index(line[6]), 3)
    ), axis=0)
    return temp


def normalize_imgs(imgs_arr, user_imgs_path):
    user_img_paths = get_filenames(user_imgs_path, 'dcm')
    num_img_ls = []
    for user_img_path in user_img_paths:
        num_img = int(user_img_path.split('/')[-1][:-4])
        num_img_ls.append(num_img)
    num_img_ls.sort()
    # print(num_img_ls)

    dist = len(num_img_ls) / imgs_arr.shape[0]
    dataset_img_paths = []
    for i in range(imgs_arr.shape[0]):
        num_img_path = '{}/{}.dcm'.format(user_imgs_path,
                                          num_img_ls[int(i * dist)])
        dataset_img_paths.append(num_img_path)
    # print(dataset_img_paths)

    for i, dataset_img_path in enumerate(dataset_img_paths):
        ct_dcm = pydicom.dcmread(dataset_img_path)
        # print(ct_dcm.pixel_array)
        ct_img = cv2.resize(ct_dcm.pixel_array, (256, 256)) / 65535
        # print(ct_img)
        imgs_arr[i] = ct_img
    return imgs_arr


def process_data(csv_file, image_dir, output_file=None, train=True, limit_num=20, image_size=256):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    output = []
    if train:
        users_id = sorted(list(set([line[0] for line in content])))
    else:
        users_id = [line[0] for line in content]

    for user_id in users_id:
        print(user_id)
        temp = {}
        user_row = list(filter(lambda x: x[0] == user_id, content))
        user_imgs_path = '{}/{}'.format(image_dir, user_id)
        user_imgs_arr = np.zeros(
            [limit_num, image_size, image_size], dtype=np.float32)
        temp.update({
            'info': np.array([normalize_info(e) for e in user_row], np.float32),
            'image': normalize_imgs(user_imgs_arr, user_imgs_path)
        })
        output.append(temp)

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(output, f)
    else:
        return output


if __name__ == '__main__':
    # statistic('raw/train.csv')
    process_data('Data/raw/train.csv', 'Data/raw/train',
                 'Data/input/train.pickle')
    # process_data('raw/test.csv', 'raw/test', 'input/test.pickle', train=False)
