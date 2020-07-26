import time, os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# TEST_CSV = '../input/osic-pulmonary-fibrosis-progression/test.csv'
# TEST_DIR = '../input/osic-pulmonary-fibrosis-progression/test'
# SUBMIT_CSV = 'submission.csv'
# MODEL_FILE = ''

TEST_CSV = 'raw/test.csv'
TEST_DIR = 'raw/test'
SUBMIT_CSV = 'output/notebook.csv'
MODEL_FILE = 'model/test_03/e35_v180.5.pickle'


# utils.py
WEEK    = (31.861846352485475, 23.240045178171002)
FVC     = (2690.479018721756, 832.5021066817238)
PERCENT = (77.67265350296326, 19.81686156299212)
AGE     = (67.18850871530019, 7.055116199848975)
IMAGE   = (615.48615, 483.8854)

def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


class Codec:
    def __init__(self, tag='fvc'):
        if tag == 'week':
            self.mean, self.std = WEEK
        elif tag == 'fvc':
            self.mean, self.std = FVC
        elif tag == 'percent':
            self.mean, self.std = PERCENT
        elif tag == 'age':
            self.mean, self.std = AGE
        elif tag == 'image':
            self.mean, self.std = IMAGE
        else:
            raise KeyError

    def encode(self, value, scale_only=False):
        value = float(value) if type(value) == str else value
        if scale_only:
            return value / self.std
        else:
            return (value - self.mean) / self.std

    def decode(self, value, scale_only=False):
        value = float(value) if type(value) == str else value
        if scale_only:
            return value * self.std
        else:
            return value * self.std + self.mean


codec_w = Codec(tag='week')
codec_f = Codec(tag='fvc')
codec_p = Codec(tag='percent')
codec_a = Codec(tag='age')
codec_i = Codec(tag='image')


# data_process.py
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
        users_id = list(set([line[0] for line in content]))
    else:
        users_id = [line[0] for line in content]

    for user_id in users_id:
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


# nets.py
class Net(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# train_osic.py
NUM_WORKERS = 4

class OsicDataset(Dataset):
    def __init__(self, arr, train=True):
        self.x = []
        self.y = []
        if train:
            for item in arr:
                for i in range(len(item['info'])):
                    for j in range(len(item['info'])):
                        temp_x = np.concatenate((
                            item['info'][i],
                            item['info'][j][0:1]
                        ), axis=0)
                        temp_y = item['info'][j][1]
                        self.x.append(temp_x)
                        self.y.append(temp_y)
        else:
            self.y = None
            for item in arr:
                for i in range(len(item['info'])):
                    for j in range(-12, 134, 1):
                        temp_x = np.concatenate((
                            item['info'][i],
                            np.array([codec_w.encode(j)], np.float32)
                        ), axis=0)
                        self.x.append(temp_x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)


class OsicModel:
    def __init__(self, name='_', net=Net(), learning_rate=0.001, step_size=20, gamma=0.7):
        self.name = name
        self.epoch = 0
        self.losses = []

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('DEVICE: {}'.format(self.device))

        self.net = net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def load_checkpoint(self, checkpoint_file, weights_only=False):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if not weights_only:
            self.epoch = checkpoint['epoch'] + 1
            self.scheduler.last_epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def predict(self, test_set, batch_size=4):
        output = []
        self.net.eval()
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                # x = data[0].to(self.device, torch.float)
                x = data.to(self.device, torch.float)

                y_pred = self.net(x)
                y_pred = y_pred.detach().squeeze().cpu().numpy()
                output.extend(y_pred)

        output = np.array(output, np.float32)
        return output


# predict.py
def write_csv(patients_id, y, c, output_file):
    make_dir(output_file)
    with open(output_file, 'w') as f:
        f.write('Patient_Week,FVC,Confidence\n')

        for w in range(146):
            for i, p in enumerate(patients_id):
                f.write('{}_{},{},{}\n'.format(p, w - 12, y[i * 146 + w], c[i * 146 + w]))


def predict(test_csv, model_file, output_file):
    with open(test_csv) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    test_arr = process_data(TEST_CSV, MODEL_FILE, train=False)
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
    predict(TEST_CSV, MODEL_FILE, SUBMIT_CSV)
