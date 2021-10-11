#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-09-17
"""

import argparse
import os
import shutil
from typing import Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchcommon.optim.lr_scheduler import CosineWarmUpAnnealingLR
from tqdm import tqdm

import dataset
import model
from evaluate import ClassificationMeter


class Trainer(object):

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 max_lr: float,
                 weight_decay: float,
                 num_epochs: int,
                 image_size: Tuple[int, int],
                 output_dir: str,
                 **kwargs):
        self._data_path = data_path
        self._batch_size = batch_size
        self._max_lr = max_lr
        self._weight_decay = weight_decay
        self._num_epochs = num_epochs

        self._image_size = image_size
        self._output_dir = output_dir

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._create_dataset()
        self._create_model()
        self._create_optimizer()

    def _create_dataset(self):
        train_transform = dataset.TrainTransform(self._image_size[0], self._image_size[1])
        train_path = os.path.join(self._data_path, 'train.ds')
        train_dataset = dataset.MyDataset(train_path, train_transform)
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            num_workers=10,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        test_transform = dataset.TestTransform(self._image_size[0], self._image_size[1])
        test_path = os.path.join(self._data_path, 'test.ds')
        test_dataset = dataset.MyDataset(test_path, test_transform)
        self._test_loader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            num_workers=10,
            pin_memory=True,
        )

    def _create_model(self):
        self._model = model.Model().to(self._device)
        self._param_list = list(self._model.parameters())

    def _create_optimizer(self):
        self._optimizer = optim.AdamW(
            self._param_list,
            lr=self._max_lr,
            weight_decay=self._weight_decay,
        )
        num_loops = self._num_epochs * len(self._train_loader)
        self._scheduler = CosineWarmUpAnnealingLR(self._optimizer, num_loops)

    def train(self):
        loss_g = None
        for epoch in range(self._num_epochs):
            self._model.train()
            loop = tqdm(self._train_loader, dynamic_ncols=True, leave=False)
            for doc in loop:
                # todo: example code
                x = doc['image']
                y = doc['label']
                loss, lr = self._train_step(x, y)

                loss = float(loss.numpy())
                loss_g = 0.99 * loss_g + 0.01 * loss if loss_g is not None else loss
                loop.set_description(f'[{epoch + 1}/{self._num_epochs}] L={loss_g:.06f} lr={lr:.01e}', False)

            self._model.eval()
            # todo: example code
            meter = ClassificationMeter(num_classes=2)
            loop = tqdm(self._test_loader, dynamic_ncols=True, leave=False)
            for doc in loop:
                x = doc['image']
                y = doc['label']
                y_ = self._model(x)
                meter.update(y_, y)
            loop.write(
                f'[{epoch + 1}/{self._num_epochs}] '
                f'Acc={meter.accuracy_score():.02%}'
            )

    def _predict_step(self, x: torch.Tensor):
        # todo: example code
        x = x.to(self._device)
        y = self._model(x)
        return y.detach().cpu()

    def _train_step(self, x: torch.Tensor, y: torch.Tensor):
        # todo: example code
        x = x.to(self._device)
        y_ = self._model(x)
        loss = (y - y_).square().mean()
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use.')
    parser.add_argument('--data-path', required=True, help='Path of the directory that contains the data files.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--max-lr', type=float, default=2e-4, help='The maximum value of learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.2, help='The weight decay value.')
    parser.add_argument('--num-epochs', type=int, default=100, help='The number of epochs to train.')
    parser.add_argument('--image-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--note')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.output_dir is not None:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    kwargs = {
        name: getattr(args, name)
        for name in dir(args)
        if not name.startswith('_')
    }
    Trainer(**kwargs).train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
