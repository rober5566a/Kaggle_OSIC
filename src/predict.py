import os, pickle
import numpy as np

from train.train_osic import OsicModel, OsicDataset
from train.nets import *
from train.utils import *


def write_csv(patients_id, y, c, output_file):
    make_dir(output_file)
    with open(output_file, 'w') as f:
        f.write('Patient_Week,FVC,Confidence\n')

        for w in range(146):
            for i, p in enumerate(patients_id):
                f.write('{}_{},{},{}\n'.format(p, w - 12, y[i * 146 + w], c[i * 146 + w]))


def predict(test_pickle, test_csv, model_file, output_file):
    with open(test_csv) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    with open(test_pickle, 'rb') as f:
        test_arr = pickle.load(f)

    test_set = OsicDataset(test_arr, train=False)

    model = OsicModel(net=Net(output_dim=2))
    model.load_checkpoint(model_file)

    patients_id = [e[0] for e in content]

    y_pred = model.predict(test_set, batch_size=16)
    y = codec_f.decode(y_pred[:, 0])
    c = codec_f.decode(y_pred[:, 1])
    # c = np.ones((len(content) * 146), np.int16) * 70

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict('input/test.pickle', 'raw/test.csv', 'model/test_03/e35_v180.5.pickle', 'output/test_03.csv')
