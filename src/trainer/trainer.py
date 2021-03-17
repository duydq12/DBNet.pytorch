# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import os
import pathlib
import shutil
import time
from pprint import pformat

import anyconfig
import torch
import torchvision.utils as vutils
from tqdm import tqdm

from src.utils import WarmupPolyLR, RunningScore, cal_text_score
from src.utils import setup_logger


class BaseTrainer:
    def __init__(self, config, model, criterion):
        config['trainer']['output_dir'] = os.path.join(
            str(pathlib.Path(os.path.abspath(__name__)).parent),
            config['trainer']['output_dir'])
        config['name'] = config['name'] + '_' + model.name
        self.save_dir = os.path.join(config['trainer']['output_dir'], config['name'])
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')

        if config['trainer']['resume_checkpoint'] == '' \
                and config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion
        # logger and tensorboard
        self.tensorboard_enable = self.config['trainer']['tensorboard']
        self.epochs = self.config['trainer']['epochs']
        self.log_iter = self.config['trainer']['log_iter']
        if config['local_rank'] == 0:
            anyconfig.dump(config, os.path.join(self.save_dir, 'config.yaml'))
            self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))
            self.logger_info(pformat(self.config))

        # device
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config['trainer']['seed'])  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config['trainer']['seed'])  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")
        self.logger_info(
            'train with device {} and pytorch {}'.format(self.device, torch.__version__))
        # metrics
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'),
                        'best_model_epoch': 0}

        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())

        # resume or finetune
        if self.config['trainer']['resume_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)

        if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
            self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler,
                                              self.optimizer)

        self.model.to(self.device)
        self.criterion.to(self.device)

        if self.tensorboard_enable and config['local_rank'] == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
            try:
                # add graph
                in_channels = 3 if config['dataset']['train']['dataset']['args'][
                                       'img_mode'] != 'GRAY' else 1
                dummy_input = torch.zeros(1, in_channels, 640, 640).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
                torch.cuda.empty_cache()
            except:
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warn('add graph to tensorboard failed')
        # 分布式训练
        if torch.cuda.device_count() > 1:
            local_rank = config['local_rank']
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   broadcast_buffers=False,
                                                                   find_unused_parameters=True)
        # make inverse Normalize
        self.UN_Normalize = False
        for t in self.config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] == 'Normalize':
                self.normalize_mean = t['args']['mean']
                self.normalize_std = t['args']['std']
                self.UN_Normalize = True

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if self.config['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            self.epoch_result = self._train_epoch(epoch)
            if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
                self.scheduler.step()
            self._on_epoch_finish()
        if self.config['local_rank'] == 0 and self.tensorboard_enable:
            self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self, epoch):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.module.state_dict() if self.config[
            'distributed'] else self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger_info(
                "resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger_info("finetune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in
                    kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def inverse_normalize(self, batch_img):
        if self.UN_Normalize:
            batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * self.normalize_std[0] + \
                                    self.normalize_mean[0]
            batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * self.normalize_std[1] + \
                                    self.normalize_mean[1]
            batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * self.normalize_std[2] + \
                                    self.normalize_mean[2]

    def logger_info(self, s):
        if self.config['local_rank'] == 0:
            self.logger.info(s)


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, metric_cls,
                 post_process=None):
        super(Trainer, self).__init__(config, model, criterion)
        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)
        if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
            if self.start_epoch > 1:
                self.config['lr_scheduler']['args']['last_epoch'] = (
                                                                            self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer,
                                          max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters,
                                          **config['lr_scheduler']['args'])
        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len,
                    len(self.validate_loader.dataset), len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(
                len(self.train_loader.dataset), self.train_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = RunningScore(2)
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]

            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            # backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()
            # acc iou
            score_shrink_map = cal_text_score(preds[:, 0, :, :], batch['shrink_map'],
                                              batch['shrink_mask'], running_metric_text,
                                              thred=self.config['post_processing']['args'][
                                                  'thresh'])

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_shrink_map: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.log_iter * cur_batch_size / batch_time, acc,
                        iou_shrink_map, loss_str, lr, batch_time))
                batch_start = time.time()

            if self.tensorboard_enable and self.config['local_rank'] == 0:
                # write tensorboard
                for key, value in loss_dict.items():
                    self.writer.add_scalar('TRAIN/LOSS/{}'.format(key), value, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_shrink_map', iou_shrink_map,
                                       self.global_step)
                self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
                if self.global_step % self.show_images_iter == 0:
                    # show images on tensorboard
                    self.inverse_normalize(batch['img'])
                    self.writer.add_images('TRAIN/imgs', batch['img'], self.global_step)
                    # shrink_labels and threshold_labels
                    shrink_labels = batch['shrink_map']
                    threshold_labels = batch['threshold_map']
                    shrink_labels[shrink_labels <= 0.5] = 0
                    shrink_labels[shrink_labels > 0.5] = 1
                    show_label = torch.cat([shrink_labels, threshold_labels])
                    show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size,
                                                  normalize=False, padding=20, pad_value=1)
                    self.writer.add_image('TRAIN/gt', show_label, self.global_step)
                    # model output
                    show_pred = []
                    for kk in range(preds.shape[1]):
                        show_pred.append(preds[:, kk, :, :])
                    show_pred = torch.cat(show_pred)
                    show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size,
                                                 normalize=False, padding=20, pad_value=1)
                    self.writer.add_image('TRAIN/preds', show_pred, self.global_step)
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr,
                'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader),
                             desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds,
                                                  is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'],
            self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False
            if self.validate_loader is not None and self.metric_cls is not None:  # 使用f1作为最优模型指标
                recall, precision, hmean = self._eval(self.epoch_result['epoch'])

                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/recall', recall, self.global_step)
                    self.writer.add_scalar('EVAL/precision', precision, self.global_step)
                    self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
                self.logger_info(
                    'test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision,
                                                                                 hmean))

                if hmean >= self.metrics['hmean']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['hmean'] = hmean
                    self.metrics['precision'] = precision
                    self.metrics['recall'] = recall
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            else:
                if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            best_str = 'current best, '
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
            self.logger_info(best_str)
            if save_best:
                import shutil
                shutil.copy(net_save_path, net_save_path_best)
                self.logger_info("Saving current best: {}".format(net_save_path_best))
            else:
                self.logger_info("Saving checkpoint: {}".format(net_save_path))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')
