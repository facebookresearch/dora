# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time

from julius import resample_frac
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dora import distrib
from .norm import SampleMEGScaler, ScaleRejectDataset, collate_rejected
from .utils import (bold, cache_name, cached_call, copy_state, pull_metric,
                    serialize_model, LogProgress)

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, datasets, model, optimizer, args):
        self.datasets = datasets
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Scaler
        self.scaler = None

        # Checkpoints
        self.continue_from = args.continue_from
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self._reset()

    def _serialize(self):
        """
        Save checkpoint.
        """
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['scaler'] = self.scaler
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        th.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        th.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

    def _reset(self):
        """
        Resume from checkpoint
        """
        load_from = None
        load_best = False
        keep_history = True
        if not self.restart:
            if self.checkpoint and self.checkpoint_file.exists():
                load_from = self.checkpoint_file
            elif self.continue_from:
                load_from = self.continue_from
                load_best = self.args.continue_best
                keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = th.load(load_from, 'cpu')
            if load_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if keep_history:
                self.history = package['history']
            self.scaler = package['scaler']
            self.best_state = package['best_state']

    def _init_scaler(self):
        if distrib.rank == 0:
            args = self.args
            if args.cache is not None:
                kwargs = dict(args.norm.scaler)
                kwargs.update(dict(args.dset))
                path = cache_name(args.cache, "scaler", kwargs)
                self.scaler = cached_call(path, self._fit_scaler)
            else:
                self.scaler = self._fit_scaler()
        self.scaler = distrib.share(self.scaler)

    def _fit_scaler(self):
        logger.info("Fitting scaler")
        loader = DataLoader(
            self.datasets['train'], shuffle=True, batch_size=10, num_workers=self.args.num_workers)
        scaler = SampleMEGScaler(**self.args.norm.scaler, device=self.device)
        scaler.fit(loader)
        return scaler

    def _init_loader(self):
        train = ScaleRejectDataset(self.datasets['train'], self.scaler, self.args.norm.max_scale)
        test = ScaleRejectDataset(self.datasets['test'], self.scaler, self.args.norm.max_scale)
        kwargs = {
            'batch_size': self.args.batch_size,
            'num_workers': self.args.num_workers,
            'collate_fn': collate_rejected,
        }
        self.train_loader = distrib.loader(train, shuffle=True, **kwargs)
        self.test_loader = distrib.loader(test, **kwargs)

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        if self.scaler is None:
            self._init_scaler()
        self._init_loader()

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                     f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()
            with th.no_grad():
                valid_loss = self._run_one_epoch(epoch, cross_valid=True)
            logger.info(
                bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                     f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())

    def _run_one_epoch(self, epoch, cross_valid=False):
        args = self.args
        task = args.task
        data_loader = self.train_loader if not cross_valid else self.test_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        total_loss = 0
        for idx, batch in enumerate(logprog):
            if batch is None:
                # All the samples are invalid, really unlikely but who knows
                continue
            meg, forcings, subjects = [x.to(self.device) for x in batch]
            sample_rate = args.sample_rate
            if task.downsample:
                meg = resample_frac(meg, sample_rate, task.downsample)
                forcings = resample_frac(forcings, sample_rate, task.downsample)
                sample_rate = task.downsample

            if task.type == "decode":
                input_ = forcings
                output = meg
            elif task.type == "encode":
                length = meg.shape[-1]
                mask = th.zeros(length).to(meg)
                limit = int(task.meg_init * sample_rate)
                mask[:limit] = 1
                input_ = th.cat([mask * meg, forcings], dim=1)
                output = meg

            if task.resample:
                input_ = resample_frac(input_, args.sample_rate, task.resample)
            estimate = self.dmodel(input_, subjects)
            if args.task.resample:
                estimate = resample_frac(estimate, args.task.resample, args.sample_rate)

            if args.task.type == "encode":
                estimate = estimate[:, :, limit:]
                output = output[:, :, limit:]
            if self.args.loss == 'l1':
                loss = F.l1_loss(output[..., :estimate.shape[-1]], estimate)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")

            # optimize model in training mode
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (idx + 1), ".5f"))
            # Just in case, clear some memory
            del loss, estimate
        return distrib.average([total_loss / (idx + 1)], idx + 1)[0]
