import argparse, os

import torch


from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class


cpu_num = 32  # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)


def main():
    # parsing configuration
    args = argparse.ArgumentParser()

    args.add_argument('-s', '--session_name', default=None,  type=str,  help="Name of training session (default: configuration file name)")
    args.add_argument('-c', '--config',       default=None,  type=str,  help="Configuration file name. (only file name in ./conf, w/o '.yaml') ")
    args.add_argument('-g', '--gpu',          default='0',   type=str,  help="GPU ID(number). Only support single gpu setting.")
    args.add_argument(      '--pretrained',   default=None,  type=str,  help="(optional)  Explicit directory of pre-trained model in ckpt folder.")
    args.add_argument('-e', '--ckpt_epoch',   default=128,   type=int,  help="Epoch number of checkpoint. (disabled when --pretrained is on)")
    args.add_argument(      '--thread',       default=4,     type=int,  help="(optional)  Number of thread for dataloader. (default: 4)")
    args.add_argument(      '--test_img',     default=None,  type=str,  help="(optional)  Image directory to denoise a single image. (default: test dataset in config file)")
    args.add_argument(      '--test_dir',     default=None,  type=str,  help="(optional)  Directory of images to denoise.")
    args.add_argument(      '--self_en',      action='store_true',      help="")
    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config  # set session name to config file name

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    # test
    trainer.test()


if __name__ == '__main__':
    main()
