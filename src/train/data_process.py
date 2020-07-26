import os, pickle
import numpy as np

from utils import *


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
        np.array([codec_w.encode(line[1]), codec_f.encode(line[2]), codec_p.encode(line[3]), codec_a.encode(line[4])], np.float32),
        onehot(['Male', 'Female'].index(line[5]), 2),
        onehot(['Never smoked', 'Ex-smoker', 'Currently smokes'].index(line[6]), 3)
    ), axis=0)
    return temp


def process_data(csv_file, image_dir, output_file=None, train=True, limit_num=20, image_size=256):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    output = []
    if train:
        users_id = sorted(list(set([line[0] for line in content])))[:160]
    else:
        users_id = [line[0] for line in content]

    for user_id in users_id:
        print(user_id)
        temp = {}
        user_row = list(filter(lambda x: x[0] == user_id, content))
        temp.update({
            'info': np.array([normalize_info(e) for e in user_row], np.float32)
        })
        output.append(temp)

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(output, f)
    else:
        return output


if __name__ == '__main__':
    # statistic('raw/train.csv')
    process_data('raw/train.csv', 'raw/train', 'input/train.pickle')
    # process_data('raw/test.csv', 'raw/test', 'input/test.pickle', train=False)
