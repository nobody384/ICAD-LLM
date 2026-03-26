import os
import torch
import random
import logging
import numpy as np


def set_environment(args):
    result_path = f'runs/{args.task_name}'
    # set output path
    if not os.path.exists(f'runs/{args.task_name}'):
        os.makedirs(result_path)

    args.output_path = os.path.join(result_path, args.data_type, args.data)

    base, index = args.output_path, 1
    while os.path.exists(base):
        base = args.output_path + '_' + str(index)
        index = index + 1
    args.output_path = base

    os.makedirs(args.output_path)

    # set logger
    log = logging.getLogger()
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s] [Line %(lineno)d] %(message)s'
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    log.setLevel(logging.INFO)

    if True:
        file_path = os.path.join(args.output_path, 'train.log')
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(log_format)
        log.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    log.addHandler(stream_handler)

    logging.info("******************** CONFIGURATION ********************")
    max_len = max([len(k) for k in vars(args).keys()]) + 4
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (max_len - len(key)))
        logging.info("%s -->    %s", keystr, val)
    logging.info("******************** CONFIGURATION ********************")

    if not torch.cuda.is_available():
        args.device = 'cpu'
        logging.info('CUDA is not available. Changing device to cpu!')

    # set random_seed
    if args.random_seed > 0:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        random.seed(args.random_seed)