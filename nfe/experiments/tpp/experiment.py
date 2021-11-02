import numpy as np
import torch

from nfe.experiments import BaseExperiment

from nfe.experiments.tpp.data import get_data_loaders
from nfe.experiments.tpp.model import JumpODE, JumpFlow, MixtureTPP, MixtureFlowTPP, MarkedTPP


class TPP(BaseExperiment):
    def get_model(self, args):
        if args.model == 'rnn':
            model = MixtureTPP(args, self.n_classes)
        elif args.decoder == 'mixture':
            model = MixtureFlowTPP(args, self.n_classes)
        elif args.model == 'ode':
            model = JumpODE(args, self.n_classes)
        elif args.model == 'flow':
            model = JumpFlow(args, self.n_classes)
        else:
            raise NotImplementedError

        self.marked_tpp = (args.marks == 1)
        if self.marked_tpp:
            model = MarkedTPP(model, self.n_classes, args.hidden_dim)

        return model.to(self.device)

    def get_data(self, args):
        return get_data_loaders(args.data, args.batch_size, self.device)

    def _get_loss(self, batch):
        times, marks, mask = batch
        times = torch.log(times + 1)
        if self.marked_tpp:
            time_loss, mark_loss = self.model(times, marks, mask)
            time_loss += times.sum() / mask.sum()
            loss = time_loss + mark_loss
        else:
            loss = self.model(times, marks, mask)[0] + times.sum() / mask.sum()
        return loss

    def training_step(self, batch):
        return self._get_loss(batch)

    def _get_dl_loss(self, dl):
        losses = []
        for batch in dl:
            losses.append(self._get_loss(batch).item())
        return np.mean(losses)

    def validation_step(self):
        return self._get_dl_loss(self.dlval)

    def test_step(self):
        return self._get_dl_loss(self.dltest)

    def finish(self):
        pass
        # OUT_DIR = ...
        # torch.save(self.model.state_dict(), OUT_DIR / 'model.pt')
