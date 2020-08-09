import time
import os
import sys
import pickle
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from nets import *
from utils import *


NUM_WORKERS = 8


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

        self.lsm_model = None
        self.net = net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

    def print_model(self):
        print(self.net)
        print('total trainable parameters: {}'.format(sum(p.numel()
                                                          for p in self.net.parameters() if p.requires_grad)))

    def save_checkpoint(self, output_file, weights_only=False):
        checkpoint = {
            'epoch': self.epoch,
            'losses': self.losses,
            'net_state_dict': self.net.state_dict(),
        }
        if not weights_only:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict()
            })
        make_dir(output_file)
        torch.save(checkpoint, output_file)

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
                # x = data[0].to(self.device, torch.float)
                x = data.to(self.device, torch.float)

                y_pred = self.net(x)
                y_pred = y_pred.detach().squeeze().cpu().numpy()
                output.extend(y_pred)

        output = np.array(output, np.float32)
        return output

    def pinball_loss(self, y_pred, y_true):
        y_true = torch.stack((y_true, y_true, y_true), dim=1)

        q = torch.FloatTensor([0.2, 0.5, 0.8]).to(self.device)
        e = y_true - y_pred

        value = torch.max(q * e, (q - 1) * e)
        value = torch.mean(value)
        return value

    def laplace_log_likelihood(self, y_pred, y_true):
        ones = torch.ones(y_pred[:, 0].size()).to(self.device)

        delta = torch.abs(y_pred[:, 0] - y_true)
        delta = torch.min(delta, ones * 1000)

        sigma = y_pred[:, 2] - y_pred[:, 1]
        sigma = torch.max(sigma, ones * 70)

        metric = math.sqrt(2) * delta / sigma + torch.log(math.sqrt(2) * sigma)
        metric = torch.mean(metric)
        return metric

    def train_on_epoch(self, loader, alpha=0.8):
        self.net.train()
        loss = 0.
        norm = 0.
        metric = 0.
        for _, data in enumerate(loader):
            self.optimizer.zero_grad()

            x = data[0].to(self.device, torch.float)
            image = data[1].to(self.device, torch.float)
            y = data[2].to(self.device, torch.float)

            y_pred = self.net(x, image)
            if self.lsm_model is not None:
                offset = self.lsm_model.forward(x).to(self.device)
                y_pred = y_pred + offset

            y_true = codec_f.decode(y)
            y_pred = codec_f.decode(y_pred)

            # loss_01 = self.pinball_loss(y_pred, y_true)
            # loss_01 = F.mse_loss(y_pred[:, 0], y_true)
            loss_02 = self.laplace_log_likelihood(y_pred, y_true)

            # print(loss_01.item(), loss_02.item())

            # batch_loss = alpha * loss_01 + (1. - alpha) * loss_02
            batch_loss = loss_02
            batch_loss.backward()

            self.optimizer.step()
            loss += batch_loss.item()
            norm += F.l1_loss(y_pred[:, 0], y_true).item()
            metric -= loss_02.item()

        return loss / len(loader), norm / len(loader), metric / len(loader)

    def val_on_epoch(self, loader, alpha=0.8):
        self.net.eval()
        with torch.no_grad():
            loss = 0.
            norm = 0.
            metric = 0.
            for _, data in enumerate(loader):
                x = data[0].to(self.device, torch.float)
                image = data[1].to(self.device, torch.float)
                y = data[2].to(self.device, torch.float)

                y_pred = self.net(x, image)
                if self.lsm_model is not None:
                    offset = self.lsm_model.forward(x).to(self.device)
                    y_pred = y_pred + offset

                y_true = codec_f.decode(y)
                y_pred = codec_f.decode(y_pred)

                # loss_01 = self.pinball_loss(y_pred, y_true)
                # loss_01 = F.mse_loss(y_pred[:, 0], y_true)
                loss_02 = self.laplace_log_likelihood(y_pred, y_true)

                # batch_loss = alpha * loss_01 * (1. - alpha) * loss_02
                batch_loss = loss_02

                loss += batch_loss
                norm += F.l1_loss(y_pred[:, 0], y_true).item()
                metric -= loss_02.item()

        return loss / len(loader), norm / len(loader), metric / len(loader)

    def fit(self, train_set, val_set=None, epochs=1, batch_size=32, checkpoint=False, save_progress=False, random_seed=None, final_model=False):
        def routine():
            print('epoch {:>3} [{:2.2f} s] '.format(
                self.epoch + 1, time.time() - start_time), end='')

            if validate:
                print('training [loss: {:3.7f}, norm: {:1.5f}, metric: {:1.5f}], validation [loss: {:3.7f}, norm: {:1.5f}, metric: {:1.5f}]'.format(
                    loss, norm, metric, val_loss, val_norm, val_metric
                ))
                if save_progress:
                    self.losses.append((loss, norm, val_loss, val_norm))
                if checkpoint and (self.epoch + 1) % checkpoint == 0:
                    folder = './output/model/{}'.format(self.name)
                    make_dir(folder)
                    if final_model:
                        self.save_checkpoint(
                            '{}/e{:02}.pickle'.format(folder, self.epoch + 1), weights_only=True)
                    else:
                        self.save_checkpoint(
                            '{}/e{:02}_v{:.1f}.pickle'.format(folder, self.epoch + 1, codec_f.decode(val_norm, True)))
            else:
                print('training [loss: {:3.7f}, norm: {:1.5f}, metric: {:1.5f}]'.format(
                    loss, norm, metric
                ))
                if save_progress:
                    self.losses.append((loss, norm))
                if checkpoint and (self.epoch + 1) % checkpoint == 0:
                    folder = './model/{}'.format(self.name)
                    make_dir(folder)
                    if final_model:
                        self.save_checkpoint(
                            '{}/e{:02}.pickle'.format(folder, self.epoch + 1), weights_only=True)
                    else:
                        self.save_checkpoint(
                            '{}/e{:02}_t{:.2f}.pickle'.format(folder, self.epoch + 1, codec_f.decode(norm, True)))

        validate = True if val_set is not None else False
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if str(self.device) == 'cuda:0':
                torch.cuda.manual_seed_all(random_seed)

        if validate:
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
            print('training on {} samples, validating on {} samples\n'.format(
                len(train_set), len(val_set)))
        else:
            print('training on {} samples\n'.format(len(train_set)))
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

        while self.epoch < epochs:
            start_time = time.time()
            loss, norm, metric = self.train_on_epoch(train_loader)
            if validate:
                val_loss, val_norm, val_metric = self.val_on_epoch(val_loader)

            routine()
            self.scheduler.step()
            self.epoch += 1


def main():
    with open('Data/input/train.pickle', 'rb') as f:
        train_arr = pickle.load(f)  # [:20]

    print(train_arr[0]['image'].shape)
    print(train_arr[0]['image'].dtype)

    with open('Data/input/val.pickle', 'rb') as f:
        val_arr = pickle.load(f)

    train_set = OsicDataset(train_arr, transform=train_transform, mode='train')
    val_set = OsicDataset(val_arr, transform=val_transform, mode='val')

    model = OsicModel('_', net=NetOI(
        input_dim=10, input_channel=1, output_dim=3), learning_rate=1e-4)
    model.fit(train_set, val_set, epochs=5, checkpoint=5, batch_size=8)


if __name__ == '__main__':
    main()
