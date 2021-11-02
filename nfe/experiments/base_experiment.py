import time
from argparse import Namespace
from copy import deepcopy
from logging import Logger
from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


class BaseExperiment:
    """ Base experiment class """
    def __init__(self, args: Namespace, logger: Logger):
        self.logger = logger
        self.args = args
        self.epochs = args.epochs
        self.patience = args.patience

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device: {self.device}')

        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest = self.get_data(args)
        self.model = self.get_model(args).to(self.device)
        logger.info(f'num_params={sum(p.numel() for p in self.model.parameters())}')

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = None
        if args.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, args.lr_scheduler_step, args.lr_decay)

    def train(self) -> None:
        # Training loop parameters
        best_loss = float('inf')
        waiting = 0
        durations = []
        best_model = deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            iteration = 0

            self.model.train()
            start_time = time.time()

            for batch in self.dltrain:
                # Single training step
                self.optim.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                ## Optional gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()

                self.logger.info(f'[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f}')
                iteration += 1

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(f'[epoch={epoch+1:04d}] epoch_duration={epoch_duration:5f}')

            # Validation step
            self.model.eval()
            val_loss = self.validation_step()
            self.logger.info(f'[epoch={epoch+1:04d}] val_loss={val_loss:.5f}')

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early stopping procedure
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model.state_dict())
                waiting = 0
            elif waiting > self.patience:
                break
            else:
                waiting += 1

        self.logger.info(f'epoch_duration_mean={np.mean(durations):.5f}')

        # Load best model
        self.model.load_state_dict(best_model)

        # Held-out test set step
        test_loss = self.test_step()
        self.logger.info(f'test_loss={test_loss:.5f}')

    def get_model(self, args: Namespace) -> Module:
        raise NotImplementedError

    def get_data(
            self,
            args: Namespace,
        ) -> Tuple[int, int, DataLoader, DataLoader, DataLoader]:
        # Returns dim, n_classes, 3 dataLoaders (train, val, test)
        raise NotImplementedError

    def training_step(self, batch: Any) -> Tensor:
        # Returns training loss (scalar)
        raise NotImplementedError

    def validation_step(self) -> Tensor:
        # Returns validation loss (scalar)
        raise NotImplementedError

    def test_step(self) -> Tensor:
        # Returns test loss (scalar)
        raise NotImplementedError

    def finish(self) -> None:
        # Performs (optional) final operations
        # e.g. save model, plot...
        return
