import os
import torch
from models import ICADLLM
import logging


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ICADLLM': ICADLLM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            if self.args.muti_gpu:
                device = 'cuda'
                logging.info('Use GPUs: {}'.format(self.args.devices))
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                logging.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            logging.info('Use CPU')
        return device

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
