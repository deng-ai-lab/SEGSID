import argparse
import sys
import os
import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import numpy as np
import random
from tqdm import tqdm

from kd.kd_dataset import SonarDataset
from kd.vqvae import VQVAE
from kd.scheduler import CycleScheduler, CosineLR
import kd.distributed as dist


def train(epoch, loader, model, optimizer, file_hander, device, checkpoint_path):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25
    mse_sum = 0
    mse_n = 0

    print_string = ''

    for i, (img, label) in enumerate(loader):
        model.train()
        model.zero_grad()

        img = img.to(device)
        label = label.to(device)
        out, latent_loss = model(img)
        recon_loss = criterion(out, label)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()
        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.9f}; "
                    f"lr: {lr:.5f}"
                )
            )
            print_string = (f"epoch: {epoch + 1}\t avg mse: {mse_sum / mse_n:.8f}\t lr: {lr:.9f}" + "\n")

            if i % 100 == 0:  # Images sampled from Training Set
                model.eval()
                sample = img[:sample_size]
                gt = label[:sample_size]
                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out, gt], 0),
                    os.path.join(checkpoint_path, 'samples', f"{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png"),
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                torch.cuda.empty_cache()

    file_hander.write(print_string)
    file_hander.flush()


def main(args):
    device = f"cuda:0"
    time_stamp = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    checkpoint_path = f'output/{args.exp_name}_stage{args.stage}_{time_stamp}'
    print(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, 'models')):
        os.mkdir(os.path.join(checkpoint_path, 'models'))
    if not os.path.exists(os.path.join(checkpoint_path, 'samples')):
        os.mkdir(os.path.join(checkpoint_path, 'samples'))
    file_hander = open(os.path.join(checkpoint_path, f'{args.exp_name}_stage{args.stage}_{time_stamp}_out.txt'), "w")
    file_hander.write(f"Exp: {args.exp_name}_stage{args.stage}_{time_stamp}" + "\n")
    file_hander.write(f'{args}' + "\n")
    if args.stage == 2:
        file_hander.write(f"Label: {args.label_dir}" + "\n")
    file_hander.flush()

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize(args.size),
                                    transforms.CenterCrop(args.size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5]), ])

    dataset = SonarDataset(input_dir=args.input_dir, label_dir=args.label_dir, transform=transform, stage=args.stage)  # if stage=2 (fine-tuning stage), label_dir shouldn't be none
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=1)

    model = VQVAE(in_channel=1).to(device)

    if args.stage == 2:
        print('Load the stage 1 mode from ', args.pretrain_model_dir)
        checkpoint = torch.load(args.pretrain_model_dir, map_location=device)
        model.load_state_dict(checkpoint)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        print('CycleScheduler is Selected')
        file_hander.write('CycleScheduler is Selected' + "\n")
        file_hander.flush()
        scheduler = CycleScheduler(optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None, warmup_proportion=0.05, )
    elif args.sched == "consine":
        print('CosineScheduler is Selected')
        file_hander.write(f'CosineScheduler is Selected\tLr max: {args.lr}\tLr min: {args.lr_min}' + "\n")
        file_hander.flush()
        scheduler = CosineLR(optimizer, lr_min=args.lr_min, lr_max=args.lr, step_size=args.epoch)

    for i in range(args.epoch):
        train(i, loader, model, optimizer, file_hander, device, checkpoint_path)
        if dist.is_primary():
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'models', f"vqvae_{str(i + 1).zfill(3)}.pt"))

        if scheduler is not None:
            scheduler.step()

    file_hander.close()


if __name__ == "__main__":
    port = (2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14 + 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",       type=str,       default=None,       help="Name of experiment session")
    parser.add_argument("--stage",          type=int,       default=1,          help="The stage of knowledge distillation, eithor 1 or 2")
    parser.add_argument("--input_dir",      type=str,       default=None,       help="Directory of input images.")
    parser.add_argument("--label_dir",      type=str,       default="",         help="Directory of target images.")
    parser.add_argument("--pretrain_model_dir", type=str,   default="",         help="Explicit directory of pre-trained model")
    parser.add_argument("--lr",             type=float,     default=3e-4,       help="Learning rate")  # LR for Stage 1
    parser.add_argument("--lr_min",         type=float,     default=5e-5,       help="Minimum Learning rate")
    parser.add_argument("--dist_url",       default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--size",           type=int,       default=160,        help="The image size")
    parser.add_argument("--epoch",          type=int,       default=560,        help="The training epoches")
    parser.add_argument("--sched",          type=str,       default='consine',  help="The learning schedule")
    parser.add_argument("--rand_seed",      type=int,       default=99,         help="")
    parser.add_argument("--n_gpu",          type=int,       default=1,          help="Only support single gpu setting.")
    parser.add_argument("--gpu",            type=str,       default='0',        help="(optional)  GPU ID(number)")
    args = parser.parse_args()


    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)  # CPU random seed
    torch.cuda.manual_seed(args.rand_seed)  # only 1 GPU
    torch.cuda.manual_seed_all(args.rand_seed)  # if >=2 GPU
    print(args)
    dist.launch(main, args.n_gpu, n_machine=1, machine_rank=0, dist_url=args.dist_url, args=(args,))
