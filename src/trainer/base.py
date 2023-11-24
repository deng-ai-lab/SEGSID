import os
import math
import time, datetime

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ..loss import Loss
from ..datahandler import get_dataset_class
from ..util.file_manager import FileManager
from ..util.logger import Logger
from ..util.util import human_format, np2tensor, rot_hflip_img, psnr, ssim, tensor2np, imread_tensor
# from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling

status_len = 13


class BaseTrainer(object):
    def test(self):
        raise NotImplementedError('define this function for each trainer')

    def validation(self):
        raise NotImplementedError('define this function for each trainer')

    def _set_module(self):
        # return dict form with model name.
        raise NotImplementedError('define this function for each trainer')

    def _set_optimizer(self):
        # return dict form with each coresponding model name.
        raise NotImplementedError('define this function for each trainer')

    def _forward_fn(self, module, loss, data):
        # forward with model, loss function and data.
        # return output of loss function.
        raise NotImplementedError('define this function for each trainer')

    # ----------------------------#
    #    Train/Test functions    #
    # ----------------------------#
    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        # get file manager and logger class
        self.file_manager = FileManager(self.session_name)
        self.logger = Logger()

        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg = cfg['validation']
        self.test_cfg = cfg['test']
        self.ckpt_cfg = cfg['checkpoint']

    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 1 and self.train_cfg['warmup']:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()

        self._after_train()

    def _warmup(self):
        self._set_status('warmup')

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.max_iter:
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' % (warmup_iter, self.max_iter))
            warmup_iter = self.max_iter

        for self.iter in range(1, warmup_iter + 1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()

    def _before_test(self, dataset_load):
        # initialing
        self.module = self._set_module()
        self._set_status('test')

        # load checkpoint file
        ckpt_epoch = self._find_last_epoch() if self.cfg['ckpt_epoch'] == -1 else self.cfg['ckpt_epoch']
        ckpt_name = self.cfg['pretrained'] if self.cfg['pretrained'] is not None else None
        self.load_checkpoint(ckpt_epoch, name=ckpt_name)
        self.epoch = self.cfg['ckpt_epoch']  # for print or saving file name.

        # test dataset loader
        if dataset_load:
            self.test_dataloader = self._set_dataloader(self.test_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # evaluation mode and set status
        self._eval_mode()
        self._set_status('test %03d' % self.epoch)

        # start message
        self.logger.highlight(self.logger.get_start_msg())

        # set denoiser
        self._set_denoiser()

        # wrapping denoiser w/ self_ensemble
        if self.cfg['self_en']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.self_ensemble(denoiser_fn, *input_data)

        # wrapping denoiser w/ crop test
        if 'crop' in self.cfg['test']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.crop_test(denoiser_fn, *input_data, size=self.cfg['test']['crop'], overlap=20)

    def _before_train(self):
        # cudnn
        torch.backends.cudnn.benchmark = False
        self._set_status('train')

        # initialing
        self.module = self._set_module()

        # training dataset loader
        self.train_dataloader = self._set_dataloader(self.train_cfg, batch_size=self.train_cfg['batch_size'], shuffle=True, num_workers=self.cfg['thread'])

        # validation dataset loader
        if self.val_cfg['val']:
            self.val_dataloader = self._set_dataloader(self.val_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # other configuration
        self.max_epoch = self.train_cfg['max_epoch']
        self.epoch = self.start_epoch = 1
        max_len = self.train_dataloader['dataset'].dataset.__len__()  # base number of iteration works for dataset named 'dataset'
        self.max_iter = math.ceil(max_len / self.train_cfg['batch_size'])

        self.loss = Loss(self.train_cfg['loss'], self.train_cfg['tmp_info'])
        self.loss_dict = {'count': 0}
        self.tmp_info = {}
        self.loss_log = []

        # set optimizer
        self.optimizer = self._set_optimizer()
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # resume
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch + 1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='w')

        # tensorboard
        tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
        self.tboard = SummaryWriter(log_dir=self.file_manager.get_dir('tboard/%s' % tboard_time))

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.info(self.summary())
        self.logger.start((self.epoch - 1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self._set_status('epoch %03d/%03d' % (self.epoch, self.max_epoch))

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        # model training mode
        self._train_mode()

    def _run_epoch(self):
        for self.iter in range(1, self.max_iter + 1):
            self._before_step()
            self._run_step()
            self._after_step()

            # if self.iter % 17== 0:
            #     break

    def _after_epoch(self):
        # save checkpoint
        if self.epoch >= self.ckpt_cfg['start_epoch']:
            if (self.epoch - self.ckpt_cfg['start_epoch']) % self.ckpt_cfg['interval_epoch'] == 0:
                self.save_checkpoint()

        # validation
        if self.val_cfg['val']:
            if self.epoch >= self.val_cfg['start_epoch'] and self.val_cfg['val']:
                if (self.epoch - self.val_cfg['start_epoch']) % self.val_cfg['interval_epoch'] == 0:
                    self._eval_mode()
                    self._set_status('val %03d' % self.epoch)
                    self.validation()
            elif self.epoch == 2 and self.val_cfg['val']:
                self._eval_mode()
                self._set_status('val %03d' % self.epoch)
                self.validation()

    def _before_step(self):
        pass

    def _run_step(self):
        # get data (data should be dictionary of Tensors)
        data = {}
        for key in self.train_dataloader_iter:
            data[key] = next(self.train_dataloader_iter[key])

        # to device
        if self.cfg['gpu'] != 'None':
            for dataset_key in data:
                for key in data[dataset_key]:
                    data[dataset_key][key] = data[dataset_key][key].cuda()

        # forward, cal losses, backward)
        losses, tmp_info = self._forward_fn(self.model, self.loss, data)
        losses = {key: losses[key].mean() for key in losses}
        tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # grad clip
        # clip_value = 10.0
        # for key in self.model:
        #     nn.utils.clip_grad_norm_(self.model[key].parameters(), max_norm=clip_value, error_if_nonfinite=True)

        # optimizer step
        for opt in self.optimizer.values():
            opt.step()

        global_iter = (self.epoch - 1) * self.max_iter + self.iter
        if global_iter % 500 == 0:
            for key in self.module:
                self.write_weight_hist(self.module[key], global_iter)

        # zero grad
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # save losses and tmp_info
        for key in losses:
            if key != 'count':
                if key in self.loss_dict:
                    self.loss_dict[key] += float(losses[key])
                else:
                    self.loss_dict[key] = float(losses[key])
        for key in tmp_info:
            if key in self.tmp_info:
                self.tmp_info[key] += float(tmp_info[key])
            else:
                self.tmp_info[key] = float(tmp_info[key])
        self.loss_dict['count'] += 1

    def _after_step(self):
        # adjust learning rate
        self._adjust_lr()

        # print loss
        if (self.iter % self.cfg['log']['interval_iter'] == 0 and self.iter != 0) or (self.iter == self.max_iter):
            self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch - 1, self.iter - 1))

    def test_dataloader_process(self, dataloader, add_con=0., floor=False, img_save=True, img_save_path=None, info=True):

        # make directory
        self.file_manager.make_dir(img_save_path)

        time_begin = time.time()
        for idx, data in enumerate(dataloader):
            # to device
            # print(data)
            if self.cfg['gpu'] != 'None':
                for key in data:
                    # print(key)
                    if key == 'file_name':
                        continue
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]

            if self.cfg['model_type'] == 'only_denoise':
                denoised_image = self.denoiser(*input_data)
            else:
                denoised_image = None
                print('error!')

            """ the range of input is  0-255 """
            denoised_image += add_con
            if floor:
                denoised_image = torch.floor(denoised_image)

            denoised_image = denoised_image.clamp(0, 255)
            noisy_img = data['noisy']

            eva_name = data['file_name'][0]

            if idx > 16:
                if hasattr(self, 'max_epoch'):
                    if self.epoch < self.max_epoch: break

            # image save
            if img_save:
                # to cpu
                if noisy_img is not None:
                    noisy_img = noisy_img.squeeze(0).cpu()
                    self.file_manager.save_img_tensor(img_save_path, '%s_N' % eva_name, noisy_img)

                denoi_img = denoised_image.squeeze(0).cpu()
                self.file_manager.save_img_tensor(img_save_path, '%s_DN' % eva_name, denoi_img)

            # procedure log msg
            if info:
                self.logger.note('[%s] testing... %04d/%04d.' % (self.status, idx, dataloader.__len__()), end='\r')

        # final log msg
        self.logger.val('[%s] Done!' % self.status)
        print('The time used is ', time.time() - time_begin)
        # return
        return None, None

    def test_img(self, image_dir, save_dir='./'):
        '''
        Inference a single image.
        '''
        # load image

        img = cv2.imread(image_dir, 1)
        img = np.average(img, axis=2, weights=[0.114, 0.587, 0.299])  # BGR -> Gray
        img = np.expand_dims(img, axis=0)  # [H,W] -> [C,H,W],C=1
        img = torch.from_numpy(np.ascontiguousarray(img).astype(np.float32))
        noisy = img.unsqueeze(0)

        # to device
        if self.cfg['gpu'] != 'None':
            noisy = noisy.cuda()

        # forward
        denoised = self.denoiser(noisy)

        # post-process
        add_con = 0. if not 'add_con' in self.test_cfg else self.test_cfg['add_con']
        floor = False if not 'floor' in self.test_cfg else self.test_cfg['floor'],

        denoised += add_con
        if floor:
            denoised = torch.floor(denoised)
        denoised = denoised.clamp(0, 255)

        # save image
        denoised = denoised.squeeze(0).cpu()
        denoised = tensor2np(denoised)

        name = image_dir.split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(save_dir, name + '_DN.png'), denoised)

        # print message
        self.logger.note('[%s] saved : %s' % (self.status, os.path.join(save_dir, name + '_DN.png')))

    def test_dir(self, direc):
        '''
        Inference all images in the directory.
        '''
        # for j in range(30):
        #     print('Check the result_dir!!!!!!!!!!!!!')
        head, tail = os.path.split(direc)
        result_dir = f'output/{tail}'
        print('Test dir and the result dir is', result_dir)
        count = 1
        time_begin_begin = time.time()

        for ff in [f for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))]:
            # print(os.listdir(direc))
            time_begin = time.time()
            # os.makedirs(os.path.join(os.path.dirname(direc), result_dir), exist_ok=True)
            # self.test_img(os.path.join(direc, ff), os.path.join(os.path.dirname(direc), result_dir))
            os.makedirs(result_dir, exist_ok=True)
            self.test_img(os.path.join(direc, ff), result_dir)
            print(f'The {count} image is {ff} \t    time used is {time.time() - time_begin}')
            count += 1

        print(f'The totally time used is {time.time() - time_begin_begin}')

    def _set_denoiser(self):
        if hasattr(self.model['denoiser'].module, 'denoise'):
            self.denoiser = self.model['denoiser'].module.denoise
        else:
            self.denoiser = self.model['denoiser'].module

    @torch.no_grad()
    def crop_test(self, fn, x, size=512, overlap=0):
        '''
        crop test image and inference due to memory problem
        '''
        b, c, h, w = x.shape

        delta = size - 2 * overlap
        if h < size:
            x = F.pad(x, (0, 0, 0, size - h), mode='reflect')
        else:
            if (h - size) % delta != 0:
                x = F.pad(x, (0, 0, 0, delta - (h - size) % delta), mode='reflect')

        if w < size:
            x = F.pad(x, (0, size - w, 0, 0), mode='reflect')
        else:
            if (w - size) % delta != 0:
                x = F.pad(x, (0, delta - (w - size) % delta, 0, 0), mode='reflect')

        _, _, new_h, new_w = x.shape
        denoised_image = torch.zeros_like(x)

        for i in range(0, new_h - delta, delta):
            for j in range(0, new_w - delta, delta):
                end_i = min(i + size, new_h)
                end_j = min(j + size, new_w)
                x_crop = x[..., i:end_i, j:end_j]
                if self.cfg['model_type'] == 'only_denoise':
                    denoised_crop = fn(x_crop)

                start_i = overlap if i != 0 else 0
                start_j = overlap if j != 0 else 0

                denoised_image[..., i + start_i:end_i, j + start_j:end_j] = denoised_crop[..., start_i:, start_j:]

        return denoised_image[:, :, :h, :w]

    @torch.no_grad()
    def self_ensemble(self, fn, x):
        '''
        Geomery self-ensemble function
        Note that in this function there is no gradient calculation.
        Args:
            fn : denoiser function
            x : input image
        Return:
            result : self-ensembled image
        '''
        result = torch.zeros_like(x)

        for i in range(8):
            tmp = fn(rot_hflip_img(x, rot_times=i % 4, hflip=i // 4))
            tmp = rot_hflip_img(tmp, rot_times=4 - i % 4)
            result += rot_hflip_img(tmp, hflip=i // 4)
        return result / 8

    # tensorboard
    def write_weight_hist(self, net, index):

        max_grad = 0
        min_grad = 0
        is_ok = True
        for name, param in net.named_parameters():
            if 'test' in name:
                continue
            try:
                if torch.max(param.grad).data > max_grad:
                    # print('name')
                    max_grad = torch.max(param.grad).data
                if torch.min(param.grad).data < min_grad:
                    min_grad = torch.min(param.grad).data

                root, sub_name = os.path.splitext(name)
                self.tboard.add_histogram(root + '/' + sub_name + '_param', param, index)
                self.tboard.add_histogram(root + '/' + sub_name + '_grad', param.grad, index)

            except Exception:
                is_ok = False
                print('name', name)
                print(f'iter-{index}-{name}_param:', param)
                print(f'iter-{index}-{name}_grad:', param.grad)
                # raise Exception('the param or param.grad ocurr NAN or INF, please check it.')

            # print('over')
        if is_ok is False:
            print('something wrong')

    # ----------------------------#
    #      Utility functions     #
    # ----------------------------#
    def print_loss(self):
        temporal_loss = 0.
        for key in self.loss_dict:
            if key != 'count':
                temporal_loss += self.loss_dict[key] / self.loss_dict['count']
        self.loss_log += [temporal_loss]
        if len(self.loss_log) > 100: self.loss_log.pop(0)

        # print status and learning rate
        loss_out_str = '[%s] %04d/%04d, lr:%s \t ' % (self.status, self.iter, self.max_iter, "{:.1e}".format(self._get_current_lr()))
        global_iter = (self.epoch - 1) * self.max_iter + self.iter

        # print losses
        avg_loss = np.mean(self.loss_log)
        loss_out_str += 'avg_100 : %.5f \t ' % (avg_loss)
        self.tboard.add_scalar('loss/avg_100', avg_loss, global_iter)

        for key in self.loss_dict:
            if key != 'count':
                loss = self.loss_dict[key] / self.loss_dict['count']
                loss_out_str += '%s : %.5f \t ' % (key, loss)
                self.tboard.add_scalar('loss/%s' % key, loss, global_iter)
                self.loss_dict[key] = 0.

        # print temporal information
        if len(self.tmp_info) > 0:
            loss_out_str += '\t['
            for key in self.tmp_info:
                loss_out_str += '  %s : %.2f' % (key, self.tmp_info[key] / self.loss_dict['count'])
                self.tmp_info[key] = 0.
            loss_out_str += ' ]'

        # reset
        self.loss_dict['count'] = 0
        self.logger.info(loss_out_str)

    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        torch.save({'epoch': self.epoch,
                    'model_weight': {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key: self.optimizer[key].state_dict() for key in self.optimizer}},
                   os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name))

    def load_checkpoint(self, load_epoch=0, name=None):
        # self._set_status('LoadCheck')
        if name is None:
            # if scratch, return
            if load_epoch == 0: return
            # load from local checkpoint folder
            file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        else:
            # load from global checkpoint folder
            file_name = os.path.join('./ckpt', name)

        print(file_name)
        # check file exist
        assert os.path.isfile(file_name), 'there is no checkpoint: %s' % file_name

        # load checkpoint (epoch, model_weight, optimizer_weight)
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.optimizer:
                self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

        # print message
        self.logger.note('[%s] model loaded : %s' % (self.status, file_name))

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d' % epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_' % self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.' % self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for first_optim in self.optimizer.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']

    def _set_dataloader(self, dataset_cfg, batch_size, shuffle, num_workers):
        dataloader = {}
        dataset_dict = dataset_cfg['dataset']
        if not isinstance(dataset_dict, dict):
            dataset_dict = {'dataset': dataset_dict}

        for key in dataset_dict:
            args = dataset_cfg[key + '_args']
            dataset = get_dataset_class(dataset_dict[key])(**args)
            dataloader[key] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

        return dataloader

    def _set_one_optimizer(self, opt, parameters, lr):
        lr = float(self.train_cfg['init_lr'])

        if opt['type'] == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt['SGD']['momentum']), weight_decay=float(opt['SGD']['weight_decay']))
        elif opt['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt['Adam']['betas'])
        elif opt['type'] == 'AdamW':
            return optim.Adam(parameters, lr=lr, betas=opt['AdamW']['betas'], weight_decay=float(opt['AdamW']['weight_decay']))
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _adjust_lr(self):
        sched = self.train_cfg['scheduler']

        if sched['type'] == 'step':
            '''
            step decreasing scheduler
            Args:
                step_size: step size(epoch) to decay the learning rate
                gamma: decay rate
            '''
            if self.iter == self.max_iter:
                args = sched['step']
                if self.epoch % args['step_size'] == 0:
                    for optimizer in self.optimizer.values():
                        lr_before = optimizer.param_groups[0]['lr']
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_before * float(args['gamma'])
        elif sched['type'] == 'linear':
            '''
            linear decreasing scheduler
            Args:
                step_size: step size(epoch) to decrease the learning rate
                gamma: decay rate for reset learning rate
            '''
            args = sched['linear']
            if not hasattr(self, 'reset_lr'):
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args['gamma']) ** ((self.epoch - 1) // args['step_size'])

            # reset lr to initial value
            if self.epoch % args['step_size'] == 0 and self.iter == self.max_iter:
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args['gamma']) ** (self.epoch // args['step_size'])
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.reset_lr
            # linear decaying
            else:
                ratio = ((self.epoch + (self.iter) / self.max_iter - 1) % args['step_size']) / args['step_size']
                curr_lr = (1 - ratio) * self.reset_lr
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

        elif sched['type'] == 'Cosine':
            if not hasattr(self, 'cos_scheduler'):
                # self.cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=True)
                self.cos_scheduler = []
                print('setting CosineAnnealingLR')
                min_lr = float(sched['min'])
                # print(min_lr,type(min_lr))
                for optimizer in self.optimizer.values():
                    self.cos_scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=min_lr, last_epoch=-1, verbose=True))

            if self.iter == self.max_iter:
                for scheduler in self.cos_scheduler:
                    scheduler.step()
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))

    def _adjust_warmup_lr(self, warmup_iter):
        init_lr = float(self.train_cfg['init_lr'])
        warmup_lr = init_lr * self.iter / warmup_iter

        for optimizer in self.optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def _train_mode(self):
        for key in self.model:
            self.model[key].train()

    def _eval_mode(self):
        for key in self.model:
            self.model[key].eval()

    def _set_status(self, status: str):
        assert len(status) <= status_len, 'status string cannot exceed %d characters, (now %d)' % (status_len, len(status))

        if len(status.split(' ')) == 2:
            s0, s1 = status.split(' ')
            self.status = '%s' % s0.rjust(status_len // 2) + ' ' \
                                                             '%s' % s1.ljust(status_len // 2)
        else:
            sp = status_len - len(status)
            self.status = ''.ljust(sp // 2) + status + ''.ljust((sp + 1) // 2)

    def summary(self):
        summary = ''

        summary += '-' * 100 + '\n'
        # model
        for k, v in self.module.items():
            # get parameter number
            param_num = sum(p.numel() for p in v.parameters())

            # get information about architecture and parameter number
            summary += '[%s] paramters: %s -->' % (k, human_format(param_num)) + '\n'
            summary += str(v) + '\n\n'

        # optim

        # Hardware

        summary += '-' * 100 + '\n'

        return summary
