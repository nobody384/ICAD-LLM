import logging

from data_provider.TimeSeries.data_factory_mixed import data_provider as timeseries_data_provider
from data_provider.TabularData.data_factory import data_provider as tabular_data_provider
from data_provider.Logs.data_factory import data_provider as log_data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
import os
import time
import warnings
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import math

warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.data_provider_dict = {
            "TimeSeries": timeseries_data_provider,
            "TabularData": tabular_data_provider,
            "Log": log_data_provider,
        }
        
        self.data_type = args.data_type
        self.normal_data_num = args.normal_data_num
        self.log_interval = args.log_interval
        self.arc_margin = args.arc_margin

        self.train_data_size = args.train_data_size
        self.val_data_size = args.val_data_size
        self.test_sample_iter = args.test_sample_iter

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.muti_gpu:
            model = torch.nn.DataParallel(model)
        return model

    def _get_data(self, flag, data_size):
        data_types = self.args.data_type.split(',')
        batch_sizes = map(int, self.args.batch_size.split(','))
        data_loaders = {}
        for data_type, batch_size in zip(data_types, batch_sizes):
            data_provider = self.data_provider_dict[data_type]
            temp_root = self.args.root_path
            self.args.root_path = os.path.join(temp_root, data_type)
            _, data_loader = data_provider(self.args, flag, data_size, batch_size)
            data_loaders[data_type] = data_loader
            self.args.root_path = temp_root
        return data_loaders

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _get_loss_func(self):
        return self._loss_func

    def _loss_func(self, y_pred_1, y_pred_2):
        return F.relu(y_pred_1 - y_pred_2 + self.arc_margin).mean()

    def random_alternate_dataloaders(self, data_loaders):
        data_pairs = [(data_type, data_loader.batch_size) for data_type, data_loader in data_loaders.items()]
        
        lcm = 1
        for _, batch_size in data_pairs:
            lcm = lcm * batch_size // math.gcd(lcm, batch_size)

        order_sequence = []
        for data_type, batch_size in data_pairs:
            logging.info(f"Data type {data_type} iterators repeat in sequence: {lcm // batch_size} times")
            order_sequence.extend([data_type] * (lcm // batch_size))
        
        iterators = {data_type: itertools.cycle(data_loader) for data_type, data_loader in data_loaders.items()}

        while True:
            for data_type in order_sequence:
                yield (next(iterators[data_type]), data_type)
    
    def vali(self, vali_loaders):
        total_loss = []
        self.model.eval()
        loss_func = self._get_loss_func()
        vali_steps = sum([len(train_loader) for _, train_loader in vali_loaders.items()])
        mixed_dataloader = self.random_alternate_dataloaders(vali_loaders)
        with torch.no_grad():
            for i, (batch_x_y, data_type) in enumerate(mixed_dataloader):
                if i > vali_steps:
                    break
                examples = batch_x_y['example'].to(self.device)
                target1 = batch_x_y['target1'].to(self.device)
                target2 = batch_x_y['target2'].to(self.device)
                outputs1 = self.model(target1, examples, data_type)
                outputs2 = self.model(target2, examples, data_type)
                loss = loss_func(outputs1, outputs2).detach().cpu()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loaders = self._get_data(flag='train', data_size=self.train_data_size)
        vali_loaders = self._get_data(flag='val', data_size=self.val_data_size)

        path = os.path.join(self.args.checkpoints, setting)
        base, index = path, 1
        while os.path.exists(base):
            base = path + '_' + str(index)
            index = index + 1
        path = base
        os.makedirs(path)

        time_now = time.time()

        train_steps = sum([len(train_loader) for _, train_loader in train_loaders.items()])
        mixed_dataloader = self.random_alternate_dataloaders(train_loaders)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs, eta_min=0)
        loss_func = self._get_loss_func()

        interval_loss_list = []

        for epoch in range(self.args.train_epochs):
            train_loss = []
            interval_loss = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x_y, data_type) in enumerate(mixed_dataloader):
                if i > train_steps:
                    break
                examples = batch_x_y['example'].to(self.device)
                target1 = batch_x_y['target1'].to(self.device)
                target2 = batch_x_y['target2'].to(self.device)
                model_optim.zero_grad()
                outputs1 = self.model(target1, examples, data_type)
                outputs2 = self.model(target2, examples, data_type)
                loss = loss_func(outputs1, outputs2)
                train_loss.append(loss.item())
                interval_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    logging.info("\tIters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, interval_loss / self.log_interval))
                    speed = (time.time() - time_now) / self.log_interval
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tSpeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()
                    interval_loss_list.append(interval_loss / self.log_interval)
                    interval_loss = 0

                loss.backward()
                model_optim.step()

            logging.info("Epoch: {} cost time: {}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            scheduler.step()
            logging.info('Updating learning rate to {}'.format(model_optim.param_groups[0]['lr']))
            
            plt.figure(figsize=(10, 6)) 
            plt.plot(interval_loss_list, label='Training Loss')
            plt.xlabel('Steps') 
            plt.ylabel('Loss') 
            plt.title('Training Loss') 
            plt.legend()  
            plt.grid(True)
            plt.savefig(os.path.join(self.args.output_path, 'train_loss.png'), format='png', dpi=300)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_loaders = self._get_data(flag='test', data_size=None)
        if test:
            logging.info('Loading model ckpt {}'.format(self.args.checkpoints))
            state_dict = torch.load(self.args.checkpoints)
            if not self.args.muti_gpu:
                logging.info("using single gpu")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_key = key.replace("module.", "")
                    new_state_dict[new_key] = value
                load_output = self.model.load_state_dict(new_state_dict, strict=False)
                print('loading:', load_output)
            else:
                self.model.load_state_dict(torch.load(self.args.checkpoints))

        self.model.eval()

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for _, test_loader in test_loaders.items():
                for i, batch_x_y in enumerate(test_loader):
                    examples = batch_x_y['example'].to(self.device)
                    target = batch_x_y['target'].to(self.device)
                    label = batch_x_y['label']
                    outputs = self.model(target, examples, self.data_type)
                    outputs = outputs.detach().cpu().numpy()
                    attens_energy.append(outputs)
                    test_labels.append(label)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.test_sample_iter).mean(axis=-1)
        test_energy = np.array(attens_energy)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1, self.test_sample_iter).mean(axis=-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        logging.info('Groud truth shape: {}'.format(gt.shape))

        logging.info('Tesing with setting: {}'.format(setting))
        def cal_F1(gt, test_energy, anomaly_ratio):
            threshold = np.percentile(test_energy, 100 - anomaly_ratio)
            print("Threshold :", threshold)
            pred = (test_energy > threshold).astype(int)

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            logging.info("Anomaly_ratio : {}".format(anomaly_ratio))
            logging.info("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            
            return accuracy, precision, recall, f_score, support

        auc_roc = roc_auc_score(gt, test_energy)

        acc, precision, recall, f_score, _ = cal_F1(gt, test_energy, self.args.anomaly_ratio)
        
        logging.info("Auc_roc : {}".format(auc_roc))

        return auc_roc, precision, recall, f_score
