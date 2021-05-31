import argparse

from dora import argparse_main, get_xp, distrib
from dora.lightning import get_trainer
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
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = F.cross_entropy(scores, y)
        acc = (y == scores.argmax(-1)).float().mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, sync_dist=True)
        self.log('valid_acc', acc, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--dummy')  # used to create multiple XP with same args.
    return parser


EXCLUDE = ['data']


@argparse_main(parser=get_parser(), dir='outputs_pl', exclude=EXCLUDE)
def main():
    args = get_xp().cfg
    world_size = distrib.get_distrib_spec().world_size
    assert args.batch_size % world_size == 0
    # Let us make sure the batch size is correct
    args.batch_size //= world_size

    data = DataModule(args.data, args.batch_size)
    module = MainModule(10)

    last = get_xp().folder / 'last.ckpt'
    resume = None
    if last.is_file():
        resume = str(last)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True)
    trainer = get_trainer(resume_from_checkpoint=resume, callbacks=[checkpoint_callback])
    trainer.fit(module, data)
