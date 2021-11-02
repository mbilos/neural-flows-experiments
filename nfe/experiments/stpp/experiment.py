import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from nfe.experiments.stpp.models.spatial import IndependentCNF, IndependentNF, SelfAttentiveCNF, SelfAttentiveNF
from nfe.experiments.stpp.toy_datasets import PinwheelHawkes
from nfe.experiments.stpp.datasets import Earthquakes, Citibike, CovidNJ
from nfe.experiments.stpp.datasets import spatiotemporal_events_collate_fn as collate_fn

from nfe.experiments import BaseExperiment


class STPP(BaseExperiment):
    def get_model(self, args):
        hidden_dims = [args.hidden_dim] * args.hidden_layers
        if args.model == 'ode':
            if args.density_model == 'independent':
                return IndependentCNF(self.dim, hidden_dims)
            elif args.density_model == 'attention':
                return SelfAttentiveCNF(self.dim, hidden_dims)
        elif args.model == 'flow':
            if args.density_model == 'independent':
                return IndependentNF(self.dim, hidden_dims, n_layers=args.flow_layers,
                    time_net=args.time_net, time_hidden_dim=args.time_hidden_dim, device=self.device)
            if args.density_model == 'attention':
                return SelfAttentiveNF(self.dim, hidden_dims, n_layers=args.flow_layers,
                    time_net=args.time_net, time_hidden_dim=args.time_hidden_dim, device=self.device)

        raise NotImplementedError

    def get_data(self, args):
        if args.data == 'pinwheel':
            dataset = PinwheelHawkes
        elif args.data == 'earthquake':
            dataset = Earthquakes
        elif args.data == 'bike':
            dataset = Citibike
        elif args.data == 'covid':
            dataset = CovidNJ
        else:
            raise NotImplementedError

        # assuming all times are normalized to (0,1) interval
        self.t0, self.t1 = torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)

        def get_dl(split):
            return DataLoader(dataset(split), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        dltrain = get_dl('train')
        dlval = get_dl('val')
        dltest = get_dl('test')
        return 2, None, dltrain, dlval, dltest

    def _get_loss(self, batch):
        t, x, m = (s.to(self.device) for s in batch)
        likelihood = self.model.logprob(t, x, m)
        loss = -(likelihood * m).sum() / m.sum()
        return loss

    def training_step(self, batch):
        return self._get_loss(batch)

    def _get_loss_for_dl(self, dl):
        losses = []
        for batch in dl:
            losses.append(self._get_loss(batch).item())
        return np.mean(losses)

    def validation_step(self):
        return self._get_loss_for_dl(self.dlval)

    def test_step(self):
        return self._get_loss_for_dl(self.dltest)

    def finish(self):
        OUT_DIR = Path('/opt/ml/model')
        if OUT_DIR.exists():
            torch.save(self.model.state_dict(), OUT_DIR / 'model.pt')
