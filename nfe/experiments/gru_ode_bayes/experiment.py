import torch
from torch.utils.data import DataLoader

from nfe.experiments.gru_ode_bayes.lib.get_model import get_gob_model
from nfe.experiments.gru_ode_bayes.lib.get_data import get_OU_data, get_MIMIC_data, get_MIMIC_data_long
from nfe.experiments.gru_ode_bayes.lib.data_utils import *
from nfe.experiments.gru_ode_bayes.lib.validate import validate

from nfe.experiments import BaseExperiment


class GOB(BaseExperiment):
    def get_model(self, args):

        self.delta_t = args.solver_step
        model = get_gob_model(self.input_size, args, self.cov_size, args.gob_p_hidden,
                              args.gob_prep_hidden, args.mixing, args.gob_cov_hidden)
        self.model = model.to(self.device)

        return model

    def get_data(self, args):
        if args.data == "2dou":
            train, val, test = get_OU_data()
        elif args.data == "mimic3" or args.data == "mimic4":
            train, val, test, value_cols = get_MIMIC_data(args.data, return_vc=True)
        else:
            raise NotImplementedError()

        dl_train = DataLoader(dataset=train, collate_fn=collate_GOB, shuffle=True, batch_size=args.batch_size)
        dl_val = DataLoader(dataset=val, collate_fn=collate_GOB, shuffle=True, batch_size=args.batch_size)
        dl_test = DataLoader(dataset=test, collate_fn=collate_GOB, shuffle=True, batch_size=args.batch_size)

        self.test_dataset = test
        self.val_dataset = val
        self.value_cols = value_cols.columns
        self.input_size = train.variable_num
        self.cov_size = train.init_cov_dim
        self.dl_val = dl_val
        self.dl_test = dl_test

        return train.variable_num, 0, dl_train, dl_val, dl_test

    def training_step(self, b):
        _, loss, _, _, _ = self.model(b['times'], b['num_obs'], b['X'].to(self.device), b['M'].to(self.device),
                                        delta_t=self.delta_t, cov=b['cov'].to(self.device), val_times=b['times_val'])
        total_loss = loss / b['y'].size(0)
        return total_loss

    def validation_step(self):
        nll, mse = validate(self.model, self.dl_val, self.device, self.delta_t)
        self.logger.info(f'val_mse={mse:.5f}')
        return nll

    def test_step(self):
        nll, mse = validate(self.model, self.dl_test, self.device, self.delta_t)
        self.logger.info(f'test_mse={mse:.5f}')
        return nll

    def eval_longer(self, args):
        val_idx = self.val_dataset.df.index.unique().tolist()
        inv_map = {v: k for k, v in self.val_dataset.map_dict.items()}
        val_idx = [inv_map[x] for x in val_idx]
        val_long = get_MIMIC_data_long(val_idx, self.value_cols, args.data)
        dl_val_long = DataLoader(dataset=val_long, collate_fn=collate_GOB,
                             shuffle=True, batch_size=args.batch_size)
        nll, mse = validate(self.model, dl_val_long, self.device, self.delta_t)
        self.logger.info(f'val_mse_long={mse:.5f}')
        self.logger.info(f'val_nll_long={nll:.5f}')

        test_idx = self.test_dataset.df.index.unique()
        inv_map = {v: k for k, v in self.test_dataset.map_dict.items()}
        test_idx = [inv_map[x] for x in test_idx]
        test_long = get_MIMIC_data_long(test_idx, self.value_cols, args.data)
        dl_test_long = DataLoader(dataset=test_long, collate_fn=collate_GOB,
                             shuffle=True, batch_size=args.batch_size)
        nll, mse = validate(self.model, dl_test_long, self.device, self.delta_t)
        self.logger.info(f'test_mse_long={mse:.5f}')
        self.logger.info(f'test_nll_long={nll:.5f}')

    def finish(self):
        pass
        # OUT_DIR = ...
        # torch.save(self.model.state_dict(), OUT_DIR / 'model.pt')
