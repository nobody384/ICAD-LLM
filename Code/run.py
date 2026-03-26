import argparse
import os
import torch
import logging
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.enviroments import set_environment


parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='AnomalyLLM')
parser.add_argument('--log_interval', type=int, default=20, help='log_interval')

# data loader
parser.add_argument('--data', type=str, required=True, default='SMD', help='dataset name')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--data_type', type=str, required=True, default='TimeSeries', help='dataset type')
parser.add_argument('--seq_len', type=int, default=7, help='seq_len')
parser.add_argument('--normal_data_num', type=int, default=3, help='number for normal data example in instruction')
parser.add_argument('--test_sample_iter', type=int, default=1, help='iter for test sample')

parser.add_argument('--train_data_size', type=float, default=100000, help='data size for traning sampling')
parser.add_argument('--val_data_size', type=float, default=100, help='data size for validation sampling')
parser.add_argument('--debug_mode', action='store_true', help='whether to load a small dataset for debugging')
parser.add_argument('--cache_time', action='store_true', help='whether to use time series data cache')
parser.add_argument('--neg_sample_rate', type=int, default=1, help='neg_sample_rate')

parser.add_argument("--sample_types", type=str, nargs="+", default=["pos", "neg"], help="List of sample types")
parser.add_argument("--sample_types_rate", type=float, nargs="+", default=[0.5, 0.5], help="List of sample type rates")
parser.add_argument('--shuffle_feature', type=int, default=1, help='status')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--gpt_layer', type=int, default=6)
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=str, default="32", help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--arc_margin', type=float, default=0.1, help='arc_margin')

# GPU
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--muti_gpu', default=False, help='using muti-gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')
parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

Exp = Exp_Anomaly_Detection

set_environment(args)

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.d_model,
            args.des, ii)

        exp = Exp(args)  # set experiments
        logging.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        logging.info('>>>>>>>end training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        torch.cuda.empty_cache()
else:
    data_list = []
    outputs = []
    setting = '{}_{}_{}_{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.d_model,
        args.des)
    if args.data_type == 'TabularData':
        data_list = [
            "Breastw",
            "Campaign",
            "Cardio",
            "Cardiotocography",
            "Fraud",
            "Glass",
            "Http",
            "Ionosphere",
            "Mammography",
            "Optdigits",
            "Pendigits",
            "Pima",
            "Satellite",
            "Satimage-2",
            "Shuttle",
            "Smtp",
            "WBC",
            "Wine",
        ]
    elif args.data_type == 'TimeSeries':
        data_list = [
            "MSL",
            "PSM",
            "SMD",
            "SMAP",
            "SWAT"
        ]
    elif args.data_type == 'Log':
        data_list = [
            'BGL',
            'Thunderbird',
            'liberty2',
            'spirit2',
        ]
    for data in data_list:
        logging.info('>>>>>>>data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(data))
        args.data = data
        exp = Exp(args)  # set experiments
        auc_roc, precision, recall, f_score = exp.test(setting, test=1)
        outputs.append(f'[{data}]  auc:{auc_roc}  precision:{precision}  recall:{recall}  f1:{f_score}')
        outputs.append(f'[{data}]  auc_roc:{auc_roc}')
    for output in outputs:
        logging.info('{}'.format(output))
    
    torch.cuda.empty_cache()
