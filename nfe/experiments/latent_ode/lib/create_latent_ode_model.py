import torch.nn as nn

from nfe.experiments.latent_ode.lib.latent_ode import LatentODE
from nfe.experiments.latent_ode.lib.encoder_decoder import *

from nfe.models import CouplingFlow, ODEModel, ResNetFlow, GRUFlow


class SolverWrapper(nn.Module):
	def __init__(self, solver):
		super().__init__()
		self.solver = solver

	def forward(self, x, t, backwards=False):
		assert len(x.shape) - len(t.shape) == 1
		t = t.unsqueeze(-1)
		if t.shape[-3] != x.shape[-3]:
			t = t.repeat_interleave(x.shape[-3], dim=-3)
		if len(x.shape) == 4:
			t = t.repeat_interleave(x.shape[0], dim=0)
		y = self.solver(x, t) # (1, batch_size, times, dim)
		return y

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
						   classif_per_tp=False, n_labels=1):

	classif_per_tp = (args.data == 'activity')

	dim = args.latents

	z0_diffeq_solver = None
	n_rec_dims = args.rec_dims
	enc_input_dim = int(input_dim) * 2 # we concatenate the mask
	gen_data_dim = input_dim

	z0_dim = args.latents
	hidden_dims = [args.hidden_dim] * args.hidden_layers

	if args.model == 'ode':
		z0_diffeq_solver = SolverWrapper(ODEModel(n_rec_dims, args.odenet, hidden_dims,
										 		  args.activation, args.final_activation, args.solver, args.solver_step,
												  args.atol, args.rtol))
		diffeq_solver = SolverWrapper(ODEModel(args.latents, args.odenet, hidden_dims,
											   args.activation, args.final_activation, args.solver, args.solver_step,
											   args.atol, args.rtol))
	elif args.model == 'flow':
		if args.flow_model == 'coupling':
			flow = CouplingFlow
		elif args.flow_model == 'resnet':
			flow = ResNetFlow
		elif args.flow_model == 'gru':
			flow = GRUFlow
		else:
			raise ValueError('Unknown flow transformation')

		z0_diffeq_solver = SolverWrapper(flow(n_rec_dims, args.flow_layers, hidden_dims, args.time_net, args.time_hidden_dim))
		diffeq_solver = SolverWrapper(flow(args.latents, args.flow_layers, hidden_dims, args.time_net, args.time_hidden_dim))
	else:
		raise NotImplementedError

	encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, z0_dim=z0_dim,
									n_gru_units=args.gru_units, device=device).to(device)

	decoder = Decoder(args.latents, gen_data_dim).to(device)

	return LatentODE(
		input_dim=gen_data_dim,
		latent_dim=args.latents,
		encoder_z0=encoder_z0,
		decoder=decoder,
		diffeq_solver=diffeq_solver,
		z0_prior=z0_prior,
		device=device,
		obsrv_std=obsrv_std,
		use_poisson_proc=False,
		use_binary_classif=args.classify,
		linear_classifier=False,
		classif_per_tp=classif_per_tp,
		n_labels=n_labels,
		train_classif_w_reconstr=(args.data == 'physionet' or args.data == 'activity')
	).to(device)
