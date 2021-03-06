#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
sys.path.append('../../')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import sliding_window, session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']
        self.criterion = None

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.sample = options['sample']
        self.feature_num = options['feature_num']
        self.num_classes = options['num_classes']
        self.early_stopping = False
        self.n_epochs_stop = options["n_epochs_stop"]
        self.epochs_no_improve = 0
        self.train_ratio = options['train_ratio']
        self.valid_ratio = options['valid_ratio']

        os.makedirs(self.save_dir, exist_ok=True)
        if self.sample == 'sliding_window':
            train_logs, train_labels = sliding_window(self.data_dir,
                                                  datatype='train',
                                                  window_size=self.window_size,
                                                  num_classes=self.num_classes,
                                                  sample_ratio=self.train_ratio
                                                      )
            val_logs, val_labels = sliding_window(self.data_dir,
                                              datatype='val',
                                              window_size=self.window_size,
                                              num_classes=self.num_classes,
                                              sample_ratio=self.valid_ratio)
        elif self.sample == 'session_window':
            train_logs, train_labels = session_window(self.data_dir,
                                                      datatype='train')
            val_logs, val_labels = session_window(self.data_dir,
                                                  datatype='val')
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics,
                                    param=self.parameters)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics,
                                    param=self.parameters)

        del train_logs
        del val_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' %
              (options['batch_size'], options['batch_size']))

        self.model = model.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        if self.sequentials:
            self.criterion = nn.CrossEntropyLoss()
        if self.parameters:
            self.criterion = nn.MSELoss()
        if self.criterion is None:
            raise NotImplementedError("train criterion is not defined")

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("\nStarting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = self.criterion
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))

            # output = self.model(features=features, device=self.device).float()
            # loss = criterion(output, label.to(self.device).float())

            output0, output1 = self.model(features=features, device=self.device)
            output0, output1 = output0.squeeze(), output1.squeeze()
            label0, label1 = label
            label0 = label0.to(self.device)
            label1 = label1.to(self.device).float()
            loss0 = nn.CrossEntropyLoss()(output0, label0)
            assert output1.shape == label1.shape
            assert output1.dtype == label1.dtype
            loss1 = nn.MSELoss()(output1, label1)
            loss = loss0 + loss1 * 1

            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

        self.log['train']['loss'].append(total_losses / num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("\nStarting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = self.criterion
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                #output = self.model(features=features, device=self.device)
                #loss = criterion(output, label.to(self.device))

                output0, output1 = self.model(features=features, device=self.device)
                output1 = output1.squeeze()
                label0, label1 = label
                label0 = label0.to(self.device)
                label1 = label1.to(self.device).float()
                loss0 = nn.CrossEntropyLoss()(output0, label0)
                loss1 = nn.MSELoss()(output1, label1)
                loss = loss0 + loss1 * 1
                total_losses += float(loss)
        print("\nValidation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve == self.n_epochs_stop:
            self.early_stopping = True
            print("Early stopping")

    def plot_train_valid_loss(self):
        train_loss = pd.read_csv(self.save_dir + "train_log.csv")
        valid_loss = pd.read_csv(self.save_dir + "valid_log.csv")
        sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
        sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
        plt.title("epoch vs train loss vs valid loss")
        plt.legend
        plt.savefig(self.save_dir + "train_valid_loss.png")
        plt.show()
        print("plot done")

    def get_error_gaussian(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.save_dir + 'deeplog_bestloss.pth')['state_dict'])
        model.eval()
        tbar = tqdm(self.valid_loader, desc="\r")
        errors = torch.tensor([], dtype=torch.float)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                # output = model(features=features, device=self.device)
                _, output = model(features=features, device=self.device)
                _, label = label
                error = output.squeeze().detach().clone().cpu().float() - label.detach().clone().cpu().float()
                errors = torch.cat((errors, error))
        std, mean = torch.std_mean(errors)
        print("The Gaussian distribution of predicted errors, --mean {:.4f} --std {:.4f}".format(mean.item(), std.item()))
        sns_plot = sns.kdeplot(errors.numpy())
        sns_plot.get_figure().savefig(self.save_dir + "valid_error_dist.png")
        plt.show()

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.early_stopping:
                break
            self.train(epoch)
            self.valid(epoch)
            self.save_log()

        if self.parameters:
            self.get_error_gaussian()

        self.plot_train_valid_loss()