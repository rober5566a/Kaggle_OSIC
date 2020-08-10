import time
import os
from os import walk
import glob
import sys
import cv2
import pickle
import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# TEST_CSV = '../input/osic-pulmonary-fibrosis-progression/test.csv'
# TEST_DIR = '../input/osic-pulmonary-fibrosis-progression/test'
# SUBMIT_CSV = 'submission.csv'
# MODEL_FILE = ''

TEST_CSV = 'Data/raw/test.csv'
TEST_DIR = 'Data/raw/test'
SUBMIT_CSV = 'output/notebook.csv'
MODEL_FILE = 'output/model/test_01/e05_v117078.5.pickle'


# utils.py
WEEK = (31.861846352485475, 23.240045178171002)
FVC = (2690.479018721756, 832.5021066817238)
PERCENT = (77.67265350296326, 19.81686156299212)
AGE = (67.18850871530019, 7.055116199848975)
IMAGE = (615.48615, 483.8854)


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

# find_file_name.py


def get_filenames(path, file_extension, isImported=False):
    file_extension = '/*.{}'.format(file_extension)
    filenames = []

    if isImported is True:
        imported_root_ls = read_imported_root_from_txt()
    else:
        imported_root_ls = []

    for root, dirs, file in walk(path):
        load_filenames = glob.glob(root + file_extension)

        if imported_root_ls == ['']:
            if len(load_filenames) == 1:
                filenames.append(load_filenames)
            else:
                filenames.extend(load_filenames)
        else:
            for filename in load_filenames:
                check_num = 0
                for imported_root in imported_root_ls:
                    if(imported_root == filename):
                        check_num = 1
                        break
                if(check_num == 1):
                    continue
                filenames.append(filename)

    return filenames


# dataset_img_process.py
def get_dataset_paths(usr_imgs_path, NUM_DIVIDED=20):
    usr_img_paths = get_filenames(usr_imgs_path, 'dcm')
    num_img_list = []
    for usr_img_path in usr_img_paths:
        num_img = int(usr_img_path.split('/')[-1][:-4])
        num_img_list.append(num_img)
    num_img_list.sort()
    # print(num_img_list)

    dist = len(num_img_list) / NUM_DIVIDED
    dataset_img_paths = []
    for i in range(NUM_DIVIDED):
        num_img_path = '{}/{}.dcm'.format(usr_imgs_path,
                                          num_img_list[int(i * dist)])
        dataset_img_paths.append(num_img_path)

    return dataset_img_paths


def remove_img_nosie(img, contours, isShow=False):
    '''
        Only save contours part, else place become back.
        ===
        create a np.zeros array(black),
        use cv2.drawContours() make contours part become 255 (white),
        final, use cv2.gitwise_and() to remove noise for img
    '''
    crop_img = np.zeros(img.shape, dtype="uint8")
    crop_img = cv2.drawContours(
        crop_img.copy(), contours, -1, 255, thickness=-1)
    crop_img = cv2.bitwise_and(img, crop_img)

    if isShow is True:
        cv2.imshow('remove_img_nosie', crop_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return crop_img


def remove_black_frame(img, contour, isShow=False):
    feature = calc_contour_feature(img, contour)
    x, y, w, h = feature[0][3]
    img_center = [int(img.shape[0]/2)+1, int(img.shape[1]/2)+1]

    if img_center[0] > y:
        w = (img_center[1] - x) * 2 - 2
        h = (img_center[0] - y) * 2 - 2
        feature[0][3] = (x, y, w, h)
    else:
        x += w
        y += h
        w = (x - (img_center[1])) * 2 - 2
        h = (y - (img_center[0])) * 2 - 2
        feature[0][3] = (2*img_center[1] - x, 2*img_center[0] - y, w, h)

    img = get_crop_img_list(
        img, feature, extra_W=-1, extra_H=-1, isShow=False)[0]
    new_img = np.ones(
        (img.shape[0]+2, img.shape[1]+2), dtype='uint8') * 255
    new_img[1:-1, 1:-1] = img

    if isShow:
        cv2.imshow('remove_black_frame', new_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return new_img


def get_biggest_countour(img, isShow=False):
    contours = get_contours_binary(img, THRESH_VALUE=100, whiteGround=False)
    new_contours = []
    contour_area_list = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > (img.size * 0.05) and contour_area < (img.size * 0.95) and contour.size > 8:
            contour_area_list.append(contour_area)
            new_contours.append(contour)

    if len(contour_area_list) != 0:
        biggest_contour = [
            new_contours[contour_area_list.index(max(contour_area_list))]]
    else:
        # need to fix : no contour fit the constrain
        biggest_contour = []
        # print(filename)

    if isShow is True:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr_img = cv2.drawContours(
            bgr_img, biggest_contour, -1, (0, 255, 0), 3)
        cv2.imshow('lung_contour', bgr_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    return biggest_contour


def get_lung_img(img, isShow=False):
    lung_contour = get_biggest_countour(img, isShow=False)

    # detect image has black frame or not
    if len(lung_contour) != 0 and np.average(img[0:10, 0:10]) < 50:
        img = remove_black_frame(img, lung_contour, isShow=False)
        lung_contour = get_biggest_countour(img)

    lung_img = remove_img_nosie(img, lung_contour, isShow=False)
    features = calc_contour_feature(lung_img, lung_contour)
    lung_img = get_crop_img_list(lung_img, features)[0]

    if isShow:
        cv2.imshow('lung_img', lung_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    return lung_img


def get_square_img(img):
    square_size = max(img.shape[:])
    square_center = int(square_size / 2)
    output_img = np.zeros(
        (square_size, square_size), dtype='uint8')
    start_point_x = square_center - int(img.shape[0]/2)
    start_point_y = square_center - int(img.shape[1]/2)
    output_img[start_point_x: start_point_x + img.shape[0],
               start_point_y: start_point_y + img.shape[1]] = img

    return output_img


# data_process.py
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


def transform_ctdata(ct_dcm, windowWidth=-1500, windowCenter=-600, CONVERT_DCM2GRAY=True):
    ct_slope = float(ct_dcm.RescaleSlope)
    ct_intercept = float(ct_dcm.RescaleIntercept)
    ct_img = ct_intercept + ct_dcm.pixel_array * ct_slope

    minWindow = - float(abs(windowCenter)) + 0.5*float(abs(windowWidth))
    new_img = (ct_img - minWindow) / float(windowWidth)
    # print(np.average(new_img))

    if np.average(new_img) > 1:  # if this img most of part are white
        try:
            minWindow = - float(abs(ct_dcm.WindowCenter)) + \
                0.5*float(abs(windowWidth))
        except TypeError:
            minWindow = - float(abs(ct_dcm.WindowCenter[0])) + \
                0.5*float(abs(windowWidth))
        new_img = (ct_img - minWindow) / float(windowWidth)

    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    if CONVERT_DCM2GRAY is True:
        new_img = (new_img * 255).astype('uint8')
    return new_img


def normalize_imgs(imgs_arr, user_imgs_path):
    user_img_paths = get_filenames(user_imgs_path, 'dcm')
    num_img_list = []
    for user_img_path in user_img_paths:
        num_img = int(user_img_path.split('/')[-1][:-4])
        num_img_list.append(num_img)
    num_img_list.sort()
    # print(num_img_list)

    dist = len(num_img_list) / imgs_arr.shape[0]
    dataset_img_paths = []
    for i in range(imgs_arr.shape[0]):
        num_img_path = '{}/{}.dcm'.format(user_imgs_path,
                                          num_img_list[int(i * dist)])
        dataset_img_paths.append(num_img_path)
    # print(dataset_img_paths)

    for i, dataset_img_path in enumerate(dataset_img_paths):
        ct_dcm = pydicom.dcmread(dataset_img_path)
        ct_img = transform_ctdata(ct_dcm, windowWidth=-1500, windowCenter=-600)
        # print(ct_dcm.pixel_array)

        j = i
        if np.average(ct_img) < 25 or np.average(ct_img) > 200 or (np.average(ct_img) == ct_img[:, :]).all():
            print('pass')
            j += 1
            ct_dcm = pydicom.dcmread(dataset_img_paths[j])
            ct_img = transform_ctdata(ct_dcm, -1500, -600)

        lung_img = get_lung_img(ct_img.copy(), isShow=False)
        output_img = get_square_img(lung_img)

        output_img = cv2.resize(
            output_img, (imgs_arr.shape[1], imgs_arr.shape[2]))
        imgs_arr[i] = output_img
    return imgs_arr


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
        user_imgs_path = '{}/{}'.format(image_dir, user_id)
        user_imgs_arr = np.zeros(
            [limit_num, image_size, image_size], dtype='uint8')
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


# Model/BoundaryDescriptor.py
def get_threshold_mask(imgray, THRESH_VALUE=170):
    ret, thresh = cv2.threshold(
        imgray, THRESH_VALUE, 255, cv2.THRESH_BINARY)  # setting threshold

    return thresh


def get_contours_binary(img, THRESH_VALUE=170, whiteGround=True, morphologyActive=False):
    if len(img.shape) > 2:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgray = img
    # print(type(imgray))

    thresh = get_threshold_mask(imgray, THRESH_VALUE)
    if whiteGround:
        thresh_white = thresh
    else:
        thresh_white = 255 - thresh

    if morphologyActive is True:
        thresh_white = cv2.morphologyEx(
            thresh_white, cv2.MORPH_OPEN, np.ones((3, 3), dtype='uint8'))
        thresh_white = cv2.morphologyEx(
            thresh_white, cv2.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'), iterations=1)

    # if your python-cv version is lower than 4.0 the cv2.findContours will return 3 variable,
    # upper 4.0 : contours, hierarchy = cv2.findContours(XXX)
    # lower 4.0 : _, contours, hierarchy = cv2.findContours(XXX)
    _, contours, hierarchy = cv2.findContours(
        thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('a', thresh_white)
    # cv2.waitKey(0)

    return contours


def calc_contour_feature(img, contours):
    feature_list = list()
    for cont in contours:
        area = cv2.contourArea(cont)
        perimeter = cv2.arcLength(cont, closed=True)
        bbox = cv2.boundingRect(cont)
        bbox2 = cv2.minAreaRect(cont)
        circle = cv2.minEnclosingCircle(cont)
        if len(cont) > 5:
            ellipes = cv2.fitEllipse(cont)
        else:
            ellipes = None
        M = cv2.moments(cont)  # return all moment of given contour
        if area != 0:  # same as M["m00"] !=0
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = (None, None)
        feature = [center, area, perimeter, bbox, bbox2, circle, ellipes]
        feature_list.append(feature)

    return feature_list


def get_crop_img_list(img, feature_list, extra_W=0, extra_H=0, isShow=False):
    crop_img_list = []
    for f in feature_list:
        (x, y, w, h) = f[3]
        x -= extra_W
        y -= extra_H
        w += extra_W*2
        h += extra_H*2

        new_position = [x, y]
        for i in range(len(new_position)):
            if new_position[i] < 0:
                new_position[i] = 0
        [x, y] = new_position

        if x + w > img.shape[1]:
            w = img.shape[1] - x
        if y + h > img.shape[0]:
            h = img.shape[0] - y

        crop_img = img[y: y + h, x: x + w]
        crop_img_list.append(crop_img)

    if isShow:
        for crop_img in crop_img_list:
            cv2.imshow("crop_img_bbox", crop_img)
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
    return crop_img_list


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


class NetOI(nn.Module):
    def __init__(self, input_dim=10, input_channel=20, output_dim=1):
        super(NetOI, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 + 10, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),

            nn.Linear(512, output_dim)
        )

    def forward(self, x, image):
        y = self.cnn(image).view(image.size(0), -1)
        # y = torch.cat([y, x[:, -1].unsqueeze(-1)], dim=1)
        y = torch.cat([y, x], dim=1)
        y = self.fc(y)

        return y


class NetSimple(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super(NetSimple, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),
#     transforms.ToTensor(),
#     transforms.RandomErasing()
# ])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# train_osic.py
NUM_WORKERS = 4


class OsicDataset(Dataset):
    def __init__(self, arr, transform, mode='train'):
        self.x = []
        self.y = []
        self.images = [item['image'] for item in arr]
        self.idx_to_image_id = []
        self.transofrm = transform
        self.mode = mode
        if mode == 'train':
            for k, item in enumerate(arr):
                for i in range(len(item['info'])):
                    for j in range(len(item['info'])):
                        temp_x = np.concatenate((
                            item['info'][i],
                            item['info'][j][0:1]
                        ), axis=0)
                        temp_y = item['info'][j][1]
                        self.x.append(temp_x)
                        self.y.append(temp_y)
                        self.idx_to_image_id.append(k)
        elif mode == 'val':
            for k, item in enumerate(arr):
                for i in range(len(item['info'])):
                    for j in [-1, -2, -3]:
                        temp_x = np.concatenate((
                            item['info'][i],
                            item['info'][j][0:1]
                        ), axis=0)
                        temp_y = item['info'][j][1]
                        self.x.append(temp_x)
                        self.y.append(temp_y)
                        self.idx_to_image_id.append(k)
        elif mode == 'test':
            self.y = None
            for k, item in enumerate(arr):
                for i in range(len(item['info'])):
                    for j in range(-12, 134, 1):
                        temp_x = np.concatenate((
                            item['info'][i],
                            np.array([codec_w.encode(j)], np.float32)
                        ), axis=0)
                        self.x.append(temp_x)
                        self.idx_to_image_id.append(k)
        else:
            raise KeyError

    def __getitem__(self, idx):
        image = torch.stack([self.transofrm(
            i) for i in self.images[self.idx_to_image_id[idx]]], dim=0).squeeze()
        # image = self.images[self.idx_to_image_id[idx]]
        image_id = random.randint(
            0, len(image) - 1) if self.mode == 'train' else len(image) // 2
        image = self.transofrm(image[image_id])
        if self.y is not None:
            return self.x[idx], image, self.y[idx]
        else:
            return self.x[idx], image

    def __len__(self):
        return len(self.x)


class OsicModel:
    def __init__(self, name='_', net=Net(), learning_rate=0.001, step_size=20, gamma=0.7):
        self.name = name
        self.epoch = 0
        self.losses = []

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('DEVICE: {}'.format(self.device))

        self.net = net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

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
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x = data[0].to(self.device, torch.float)
                image = data[1].to(self.device)

                y_pred = self.net(x, image)
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
                f.write('{}_{},{},{}\n'.format(
                    p, w - 12, y[i * 146 + w], c[i * 146 + w]))


def predict(test_csv, model_file, output_file):
    with open(test_csv) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    test_arr = process_data(TEST_CSV, TEST_DIR, train=False)
    test_set = OsicDataset(test_arr, val_transform, mode='test')

    model = OsicModel(net=NetOI(input_dim=10, input_channel=1, output_dim=3))
    model.load_checkpoint(model_file)

    patients_id = [e[0] for e in content]

    y_pred = model.predict(test_set, batch_size=16)
    y = codec_f.decode(y_pred[:, 0])
    c = codec_f.decode(y_pred[:, 2] - y_pred[:, 1], scale_only=True)
    # c = np.ones((len(content) * 146), np.int16) * 70

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict(TEST_CSV, MODEL_FILE, SUBMIT_CSV)
