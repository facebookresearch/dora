# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys

from dora import argparse_main, get_xp, distrib
from dora.lightning import trainer_from_argparse_args, PLLogProgress
from dora.log import colorize
import torch
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .data import DataModule


class MainModule(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = F.cross_entropy(scores, y)
        # Those metrics will be forwarded to Dora Link automatically.
        self.mylog('train_loss', loss, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = F.cross_entropy(scores, y)
        acc = (y == scores.argmax(-1)).float().mean()
        # Those metrics will be forwarded to Dora Link automatically.
        self.mylog('valid_loss', loss)
        self.mylog('valid_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def mylog(self, name, value, train=False):
        # Here, it is quite important to have `sync_dist` set to True for valid,
        # as otherwise you will get inacurate validation metrics (only on a shard of the data).
        # If you are using PLLogProgress, set `prog_bar=True` along with `on_step=True`.
        # If you want to metric logged to Dora link history, so that it appears in
        # grid searches table, set `on_epoch=True`. PL typically add a `_epoch` suffix
        # for epoch wise metrics. This will be automatically stripped and only the
        # epoch wise metric will be saved.
        self.log(name, value, on_epoch=True, sync_dist=not train, on_step=True, prog_bar=True)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/tmp/dora_test_mnist')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--dummy')  # used to create multiple XP with same args.
    pl.Trainer.add_argparse_args(parser)
    return parser


EXCLUDE = ['data', 'restart']


@argparse_main(parser=get_parser(), dir='outputs_pl', exclude=EXCLUDE, use_underscore=True)
def main():
    log_format = "".join([
        "[" + colorize("%(asctime)s", "36") + "]",
        "[" + colorize("%(name)s", "34") + "]",
        "[" + colorize("%(levelname)s", "32") + "]",
        " - ",
        "%(message)s",
    ])
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        datefmt="%m-%d %H:%M:%S",
        format=log_format)
    logger_prog = logging.getLogger("progress")
    progress = PLLogProgress(logger_prog, updates=10)
    args = get_xp().cfg
    world_size = distrib.get_distrib_spec().world_size
    assert args.batch_size % world_size == 0
    # Let us make sure the batch size is correct
    args.batch_size //= world_size

    data = DataModule(args.data, args.batch_size)
    module = MainModule(10)

    checkpoint_callback = ModelCheckpoint(
        dirpath=get_xp().folder, monitor='valid_loss', save_last=True)
    trainer = trainer_from_argparse_args(args, callbacks=[checkpoint_callback, progress])
    trainer.fit(module, data)
