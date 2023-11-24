import argparse, os
from importlib import import_module

import torch
import numpy as np
import random
from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class
import torch.backends.cudnn as cudnn


cpu_num = 16  # number of CPUs
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name', default=None,  type=str,  help="Name of training session (default: configuration file name)")
    args.add_argument('-c', '--config',       default=None,  type=str,  help="Configuration file name. (only file name in ./conf, w/o '.yaml') ")
    args.add_argument('-g', '--gpu',          default='0',  type=str,   help="GPU ID(number). Only support single gpu setting.")
    args.add_argument('-r', '--resume',       action='store_true',      help="(optional)  Flag for resume training. (On: resume, Off: starts from scratch)")
    args.add_argument(      '--thread',       default=16,    type=int,  help="(optional)  Number of thread for dataloader. (default: 16)")
    args.add_argument(      '--rand_seed',    default=0,  type=int,     help="(optional)  Random seed")

    args = args.parse_args()
    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config # set session name to config file name

    cfg = ConfigParser(args)
    print('Session Name: ', cfg['session_name'])

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    os.environ['PYTHONHASHSEED'] = str(cfg['rand_seed'])
    np.random.seed(cfg['rand_seed'])
    random.seed(cfg['rand_seed'])
    torch.manual_seed(cfg['rand_seed'])  # CPU random seed
    torch.cuda.manual_seed(cfg['rand_seed'])  # only 1 GPU
    torch.cuda.manual_seed_all(cfg['rand_seed'])  # if >=2 GPU

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    # train
    trainer.train()


if __name__ == '__main__':
    main()
