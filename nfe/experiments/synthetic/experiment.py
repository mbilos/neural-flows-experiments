from pathlib import Path

import numpy as np
import torch

from nfe.experiments import BaseExperiment
from nfe.experiments.synthetic.data import get_data_loaders, get_single_loader
from nfe.models import ODEModel, CouplingFlow, ResNetFlow


class Synthetic(BaseExperiment):
    def get_model(self, args):
        if args.model == 'ode':
            return ODEModel(self.dim, args.odenet, [args.hidden_dim] * args.hidden_layers, args.activation,
                            args.final_activation, args.solver, args.solver_step, args.atol, args.rtol)
        elif args.model == 'flow':
            if args.flow_model == 'coupling':
                return CouplingFlow(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim)
            elif args.flow_model == 'resnet':
                return ResNetFlow(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                  args.time_net, args.time_hidden_dim)
        raise NotImplementedError

    def get_data(self, args):
        return get_data_loaders(args.data, args.batch_size)

    def _get_loss(self, batch):
        x, t, y_true = batch
        y = self.model(x, t)
        assert y.shape == y_true.shape
        loss = torch.mean((y - y_true)**2)
        return loss

    def _get_loss_on_dl(self, dl):
        losses = []
        for batch in dl:
            losses.append(self._get_loss(batch).item())
        return np.mean(losses)

    def training_step(self, batch):
        return self._get_loss(batch)

    def validation_step(self):
        return self._get_loss_on_dl(self.dlval)

    def test_step(self):
        return self._get_loss_on_dl(self.dltest)

    def _sample_trajectories(self, path):
        N, M, T = 21, 200, 30
        x = torch.linspace(-5, 5, N).view(N, 1, 1)
        t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1)
        y = self.model(x, t)
        np.savez(path, x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=y.detach().cpu().numpy())

    def finish(self):
        dl_extrap_time = get_single_loader(f'{self.args.data}_extrap_time', self.args.batch_size)
        dl_extrap_space = get_single_loader(f'{self.args.data}_extrap_space', self.args.batch_size)

        loss_time = self._get_loss_on_dl(dl_extrap_time)
        loss_space = self._get_loss_on_dl(dl_extrap_space)

        self.logger.info(f'loss_extrap_time={loss_time:.5f}')
        self.logger.info(f'loss_extrap_space={loss_space:.5f}')

        ## Uncomment to save models
        # OUT_DIR = ...
        # torch.save(self.model.state_dict(), OUT_DIR / 'model.pt')
