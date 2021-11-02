import argparse
import logging

import numpy as np
import torch

from nfe.experiments.gru_ode_bayes.experiment import GOB
from nfe.experiments.latent_ode.experiment import LatentODE
from nfe.experiments.stpp.experiment import STPP
from nfe.experiments.synthetic.experiment import Synthetic
from nfe.experiments.tpp.experiment import TPP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser('Neural flows')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--experiment', type=str, help='Which experiment to run',
                    choices=['latent_ode', 'synthetic', 'gru_ode_bayes', 'tpp', 'stpp'])
parser.add_argument('--model', type=str, help='Whether to use ODE or flow based model or RNN',
                    choices=['ode', 'flow', 'rnn'])
parser.add_argument('--data',  type=str, help='Dataset name',
                    choices=['hopper', 'physionet', 'activity', # latent ode
                             'sine', 'square', 'triangle', 'sawtooth', 'sink', 'ellipse', # synthetic
                             'mimic3', 'mimic4', '2dou', #gru-ode-bayes
                             'hawkes1', 'hawkes2', 'poisson', 'renewal', 'reddit', 'mooc', 'lastfm', 'wiki', # tpp
                             'pinwheel', 'earthquake', 'covid', 'bike', # stpp
                    ])

# Training loop args
parser.add_argument('--epochs', type=int, default=1000, help='Max training epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay (regularization)')
parser.add_argument('--lr-scheduler-step', type=int, default=-1, help='Every how many steps to perform lr decay')
parser.add_argument('--lr-decay', type=float, default=0.9, help='Multiplicative lr decay factor')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--clip', type=float, default=1, help='Gradient clipping')

# NN args
parser.add_argument('--hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--hidden-dim', type=int, default=1, help='Size of hidden layer')
parser.add_argument('--activation', type=str, default='Tanh', help='Hidden layer activation')
parser.add_argument('--final-activation', type=str, default='Identity', help='Last layer activation')

# ODE args
parser.add_argument('--odenet', type=str, default='concat', help='Type of ODE network', choices=['concat', 'gru']) # gru only in GOB
parser.add_argument('--solver', type=str, default='dopri5', help='ODE solver', choices=['dopri5', 'rk4', 'euler'])
parser.add_argument('--solver_step', type=float, default=0.05, help='Fixed solver step')
parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance')
parser.add_argument('--rtol', type=float, default=1e-3, help='Relative tolerance')

# Flow model args
parser.add_argument('--flow-model', type=str, default='coupling', help='Model name', choices=['coupling', 'resnet', 'gru'])
parser.add_argument('--flow-layers', type=int, default=1, help='Number of flow layers')
parser.add_argument('--time-net', type=str, default='TimeLinear', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
parser.add_argument('--time-hidden-dim', type=int, default=1, help='Number of time features (only for Fourier)')

# latent_ode specific args
parser.add_argument('--classify', type=int, default=0, help='Include classification loss (physionet and activity)', choices=[0, 1])
parser.add_argument('--extrap', type=int, default=0, help='Set extrapolation mode. Else run interpolation mode.', choices=[0, 1])
parser.add_argument('-n',  type=int, default=10000, help='Size of the dataset (latent_ode)')
parser.add_argument('--quantization', type=float, default=0.016, help='Quantization on the physionet dataset.')
parser.add_argument('--latents', type=int, default=20, help='Size of the latent state')
parser.add_argument('--rec-dims', type=int, default=20, help='Dimensionality of the recognition model (ODE or RNN).')
parser.add_argument('--gru-units', type=int, default=100, help='Number of units per layer in each of GRU update networks')
parser.add_argument('--timepoints', type=int, default=100, help='Total number of time-points')
parser.add_argument('--max-t',  type=float, default=5., help='We subsample points in the interval [0, args.max_tp]')

# GRU-ODE-Bayes specific args
parser.add_argument('--mixing', type=float, default=0.0001, help='Ratio between KL and update loss')
parser.add_argument('--gob_prep_hidden', type=int, default=10, help='Size of hidden state for covariates')
parser.add_argument('--gob_cov_hidden', type=int, default=50, help='Size of hidden state for covariates')
parser.add_argument('--gob_p_hidden', type=int, default=25, help='Size of hidden state for initialization')
parser.add_argument('--invertible', type=int, default=1, help='If network is invertible', choices=[0, 1])

# TPP args
parser.add_argument('--components', type=int, default=8, help='Number of mixture components')
parser.add_argument('--decoder', type=str, default='continuous', help='Intensity function', choices=['continuous', 'mixture'])
parser.add_argument('--rnn', type=str, help='RNN encoder', choices=['gru', 'lstm'])
parser.add_argument('--marks', type=int, default=0, help='Whether to use marked TPP', choices=[0, 1])

# STPP args
parser.add_argument('--density-model', type=str, help='Type of density model', choices=['independent', 'attention', 'jump'])

args = parser.parse_args()

def get_experiment(args, logger):
    if args.experiment == 'latent_ode':
        return LatentODE(args, logger)
    elif args.experiment == 'synthetic':
        return Synthetic(args, logger)
    elif args.experiment == 'gru_ode_bayes':
        return GOB(args, logger)
    elif args.experiment == 'tpp':
        return TPP(args, logger)
    elif args.experiment == 'stpp':
        return STPP(args, logger)
    else:
        raise ValueError(f'Need to specify experiment')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment = get_experiment(args, logger)

    experiment.train()
    experiment.finish()
