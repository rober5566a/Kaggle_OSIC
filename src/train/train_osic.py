import time
import os
import sys
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from nets import *
from utils import *


NUM_WORKERS = 4


class OsicDataset(Dataset):
    def __init__(self, arr, mode='train'):
        self.x = []
        self.y = []
        if mode == 'train':
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
        elif mode == 'val':
            for item in arr:
                for i in range(len(item['info'])):
                    for j in [-1, -2, -3]:
                        temp_x = np.concatenate((
                            item['info'][i],
                            item['info'][j][0:1]
                        ), axis=0)
                        temp_y = item['info'][j][1]
                        self.x.append(temp_x)
                        self.y.append(temp_y)
        elif mode == 'test':
            self.y = None
            for item in arr:
                for i in range(len(item['info'])):
                    for j in range(-12, 134, 1):
                        temp_x = np.concatenate((
                            item['info'][i],
                            np.array([codec_w.encode(j)], np.float32)
                        ), axis=0)
                        self.x.append(temp_x)
        else:
            raise KeyError

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

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('DEVICE: {}'.format(self.device))

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
            y = data[1].to(self.device, torch.float)

            y_pred = self.net(x)

            y = codec_f.decode(y)
            y_pred = codec_f.decode(y_pred)

            loss_01 = self.pinball_loss(y_pred, y)
            loss_02 = self.laplace_log_likelihood(y_pred, y)

            batch_loss = alpha * loss_01 + (1. - alpha) * loss_02
            batch_loss.backward()

            self.optimizer.step()
            loss += batch_loss.item()
            norm += F.l1_loss(y_pred[:, 0], y).item()
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
                y = data[1].to(self.device, torch.float)

                y_pred = self.net(x)

                y = codec_f.decode(y)
                y_pred = codec_f.decode(y_pred)

                loss_01 = self.pinball_loss(y_pred, y)
                loss_02 = self.laplace_log_likelihood(y_pred, y)

                batch_loss = alpha * loss_01 * (1. - alpha) * loss_02

                loss += batch_loss
                norm += F.l1_loss(y_pred[:, 0], y).item()
                metric -= loss_02.item()

        return loss / len(loader), norm / len(loader), metric / len(loader)

    # def train_on_epoch(self, loader, alpha=0.8):
    #     self.net.train()
    #     loss = 0.
    #     norm = 0.
    #     metric = 0.
    #     for _, data in enumerate(loader):
    #         self.optimizer.zero_grad()

    #         x = data[0].to(self.device, torch.float)
    #         y = data[1].to(self.device, torch.float)

    #         y_pred = self.net(x)

    #         batch_loss_01 = F.mse_loss(y_pred[:, 0], y)

    #         delta = torch.abs(y_pred[:, 0] - y)
    #         sigma = y_pred[:, 1]

    #         delta = codec_f.decode(delta, scale_only=True)
    #         sigma = codec_f.decode(sigma)
    #         sigma = torch.clamp(sigma, 70, 5000)

    #         batch_loss_02 = torch.mean(math.sqrt(2) * delta / sigma + torch.log(math.sqrt(2) * sigma))

    #         batch_loss = 0.5 * batch_loss_01 + 0.5 * batch_loss_02
    #         batch_loss.backward()

    #         self.optimizer.step()
    #         loss += batch_loss.item()
    #         norm += F.l1_loss(y_pred[:, 0], y).item()
    #         metric -= batch_loss_02.item()

    #     return loss / len(loader), norm / len(loader), metric / len(loader)

    # def val_on_epoch(self, loader, alpha=0.8):
    #     self.net.eval()
    #     with torch.no_grad():
    #         loss = 0.
    #         norm = 0.
    #         metric = 0.
    #         for _, data in enumerate(loader):
    #             x = data[0].to(self.device, torch.float)
    #             y = data[1].to(self.device, torch.float)

    #             y_pred = self.net(x)

    #             batch_loss_01 = F.mse_loss(y_pred[:, 0], y)

    #             delta = torch.abs(y_pred[:, 0] - y)
    #             sigma = y_pred[:, 1]

    #             delta = codec_f.decode(delta, scale_only=True)
    #             sigma = codec_f.decode(sigma)
    #             sigma = torch.clamp(sigma, 70, 5000)

    #             batch_loss_02 = torch.mean(math.sqrt(2) * delta / sigma + torch.log(math.sqrt(2) * sigma))

    #             batch_loss =  0.5 * batch_loss_01 + 0.5 * batch_loss_02

    #             loss += batch_loss
    #             norm += F.l1_loss(y_pred[:, 0], y).item()
    #             metric -= batch_loss_02

    #     return loss / len(loader), norm / len(loader), metric / len(loader)

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
                    folder = './model/{}'.format(self.name)
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
    with open('input/train.pickle', 'rb') as f:
        train_arr = pickle.load(f)

    with open('input/val.pickle', 'rb') as f:
        val_arr = pickle.load(f)

    train_set = OsicDataset(train_arr, mode='train')
    val_set = OsicDataset(val_arr, mode='val')

    model = OsicModel('test_04', net=NetSimple(
        input_dim=10, output_dim=3), learning_rate=1e-4)
    model.fit(train_set, epochs=200, batch_size=128)


if __name__ == '__main__':
    main()
