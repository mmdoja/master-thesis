from core.baseengine import Basebaseengine
from pyhocon import ConfigTree
from core.iterators import DataLoaderFactory
from core.models import ModelFactory
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import argparse
import wandb
import random
from torch import nn, optim
from core.utils.torchpie import AverageMeter
import time
from core.iterators.YT_data import YoutubeDataset
from core.Loss import SmoothCrossEntropyLoss
from core.optimizer import CustomSchedule
from core.accuracy import compute_epiano_accuracy
from pprint import pprint
from pyhocon import ConfigFactory, ConfigTree


class baseengine(Basebaseengine):

    @staticmethod
    def epoch_time(start_time: float, end_time: float):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_loss = float('inf')
        for epoch in range(self.epochs_size):
            start_time = time.time()
            _train_loss = self.train(epoch)
            loss = self.test(epoch)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            torch.save(
                {
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
                os.path.join(self.experiment_path, "checkpoint.pt")
            )

    def close(self):
        self.summary_writer.close()

    def __init__(self, cfg: ConfigTree, args):
        self.cfg = cfg
        self.experiment_path = args.exps
        self.summary_writer = SummaryWriter(log_dir=self.experiment_path)
        self.model_builder = ModelFactory(cfg)
        self.dataset_builder = DataLoaderFactory(cfg)

        self.train_ds = self.dataset_builder.build(split='train')
        self.test_ds = self.dataset_builder.build(split='val')
        self.ds: YoutubeDataset = self.train_ds.dataset

        self.train_Loss = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.val_Loss = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.model: nn.Module = self.model_builder.build(device=torch.device('cuda'), wrapper=nn.DataParallel)
        optimizer = optim.Adam(self.model.parameters(), lr=0., betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = CustomSchedule(
            self.cfg.get_int('model.emb_dim'),
            optimizer=optimizer,
        )

        self.epochs_size = cfg.get_int('epochs_size')

        print(f'Use control: {self.ds.use_control}')

    wandb.init(project='genmusic', sync_tensorboard=True)


    def train(self, epoch=0):
        calc_loss = AverageMeter()
        calc_acc = AverageMeter()
        num_iters = len(self.train_ds)
        self.model.train()
        for i, data in enumerate(self.train_ds):
            midi_x, midi_y = data['midi_x'], data['midi_y']

            if self.ds.use_pose:
                feat = data['pose']
            elif self.ds.use_rgb:
                feat = data['rgb']
            elif self.ds.use_flow:
                feat = data['flow']
            else:
                raise Exception('No feature!')
            feat, midi_x, midi_y = (
                feat.cuda(non_blocking=True),
                midi_x.cuda(non_blocking=True),
                midi_y.cuda(non_blocking=True)
            )

            if self.ds.use_control:
                control = data['control']
                control = control.cuda(non_blocking=True)
            else:
                control = None
            output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

            loss = self.train_Loss(output.view(-1, output.shape[-1]), midi_y.flatten())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            acc = compute_epiano_accuracy(output, midi_y, pad_idx=self.ds.PAD_IDX)

            batch_size = len(midi_x)
            calc_loss.update(loss.item(), batch_size)
            calc_acc.update(acc.item(), batch_size)

            print(
                f'Train [{epoch}/{self.epochs_size}][{i}/{num_iters}]\t'
                f'{calc_loss.avg}\t{calc_acc.avg}'
            )
        self.summary_writer.add_scalar('train/loss', calc_loss.avg, epoch)
        self.summary_writer.add_scalar('train/acc', calc_acc.avg, epoch)

        #wandb.log({'epoch': epoch, 'train/loss': calc_loss.avg})
        #wandb.log({'epoch': epoch, 'train/acc': calc_acc.avg})

        return calc_loss.avg

    def test(self, epoch=0):
        calc_loss = AverageMeter()
        calc_acc = AverageMeter()
        num_iters = len(self.test_ds)
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.test_ds):
                midi_x, midi_y = data['midi_x'], data['midi_y']

                if self.ds.use_pose:
                    feat = data['pose']
                elif self.ds.use_rgb:
                    feat = data['rgb']
                elif self.ds.use_flow:
                    feat = data['flow']
                else:
                    raise Exception('No feature!')

                feat, midi_x, midi_y = (
                    feat.cuda(non_blocking=True),
                    midi_x.cuda(non_blocking=True),
                    midi_y.cuda(non_blocking=True)
                )
                if self.ds.use_control:
                    control = data['control']
                    control = control.cuda(non_blocking=True)
                else:
                    control = None

                output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

                loss = self.val_Loss(output.view(-1, output.shape[-1]), midi_y.flatten())

                acc = compute_epiano_accuracy(output, midi_y)

                batch_size = len(midi_x)
                calc_loss.update(loss.item(), batch_size)
                calc_acc.update(acc.item(), batch_size)
                print(
                    f'Val [{epoch}/{self.epochs_size}][{i}/{num_iters}]\t'
                    f'{calc_loss.avg}\t{calc_acc.avg}'
                )
            self.summary_writer.add_scalar('val/loss', calc_loss.avg, epoch)
            self.summary_writer.add_scalar('val/acc', calc_acc.avg, epoch)

            #wandb.log({'epoch': epoch, 'val/loss': calc_loss.avg})
            #wandb.log({'epoch': epoch, 'val/acc': calc_acc.avg})

        return calc_loss.avg

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="")
    parser.add_argument('--exps', '-e', type=str, default="")
    args = parser.parse_args()
    cfg = ConfigFactory.parse_file(args.config)
    print('=' * 100)
    pprint(cfg)
    print('=' * 100)
    baseengine = baseengine(cfg, args)
    baseengine.run()
    baseengine.close()


if __name__ == '__main__':
    main()
