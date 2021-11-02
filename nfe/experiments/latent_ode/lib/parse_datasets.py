from pathlib import Path
import numpy as np
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader

from nfe.experiments.latent_ode.mujoco_physics import HopperPhysics
from nfe.experiments.latent_ode.physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from nfe.experiments.latent_ode.person_activity import PersonActivity, variable_time_collate_fn_activity

import nfe.experiments.latent_ode.lib.utils as utils

DATA_DIR = Path('/opt/ml/input/data/training')
LOCAL = False
if not DATA_DIR.exists():
	LOCAL = True
	DATA_DIR = Path(__file__).parents[2] / 'data'

def parse_datasets(args, device):

	def basic_collate_fn(batch, time_steps, args=args, device=device, data_type='train'):
		batch = torch.stack(batch)
		data_dict = {
			'data': batch,
			'time_steps': time_steps.unsqueeze(0)
		}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict

	dataset_name = args.data
	output_dim = 1

	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp

	##################################################################
	# MuJoCo dataset
	if dataset_name == 'hopper':
		dataset_obj = HopperPhysics(root='data', download=True, generate=False, device=device)
		dataset = dataset_obj.get_dataset()[:args.n]
		dataset = dataset.to(device)

		n_tp_data = dataset[:].shape[1]

		# Time steps that are used later on for exrapolation
		time_steps = torch.arange(start=0, end=n_tp_data, step=1).float().to(device)
		time_steps = time_steps / len(time_steps)

		dataset = dataset.to(device)
		time_steps = time_steps.to(device)

		if not args.extrap:
			# Creating dataset for interpolation
			# sample time points from different parts of the timeline,
			# so that the model learns from different parts of hopper trajectory
			n_traj = len(dataset)
			n_tp_data = dataset.shape[1]
			n_reduced_tp = args.timepoints

			# sample time points from different parts of the timeline,
			# so that the model learns from different parts of hopper trajectory
			start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
			end_ind = start_ind + n_reduced_tp
			sliced = []
			for i in range(n_traj):
				  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
			dataset = torch.stack(sliced).to(device)
			time_steps = time_steps[:n_reduced_tp]

		train_y, val_y, test_y = utils.split_train_val_test(dataset)

		n_samples = len(dataset)
		input_dim = dataset.size(-1)

		dltrain = DataLoader(train_y, batch_size=args.batch_size, shuffle=True,
			collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type='train'))
		dlval = DataLoader(val_y, batch_size=args.batch_size, shuffle=False,
			collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type='test'))
		dltest = DataLoader(test_y, batch_size=args.batch_size, shuffle=False,
			collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type='test'))

	##################################################################
	# Physionet dataset
	if dataset_name == 'physionet':
		train_dataset_obj = PhysioNet(DATA_DIR / 'physionet' if LOCAL else DATA_DIR, train=True, quantization=args.quantization,
                                	  download=True, n_samples = min(10000, args.n), device=device)
		test_dataset_obj = PhysioNet(DATA_DIR / 'physionet' if LOCAL else DATA_DIR, train=False, quantization=args.quantization,
										download=True, n_samples=min(10000, args.n), device=device)

		# Combine and shuffle samples from physionet Train and physionet Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]

		if not args.classify:
			# Concatenate samples from original Train and Test sets
			# Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
			total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

		train_data, val_data, test_data = utils.split_train_val_test(total_dataset)

		n_samples = len(total_dataset)
		input_dim = total_dataset[0][2].shape[-1]

		batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
		data_min, data_max = get_data_min_max(total_dataset)

		dltrain = DataLoader(train_data, batch_size=batch_size, shuffle=True,
			collate_fn=lambda batch: variable_time_collate_fn(batch, args, device, data_type='train', data_min=data_min, data_max=data_max))
		dlval = DataLoader(val_data, batch_size=batch_size, shuffle=False,
			collate_fn=lambda batch: variable_time_collate_fn(batch, args, device, data_type='test', data_min=data_min, data_max=data_max))
		dltest = DataLoader(test_data, batch_size=batch_size, shuffle=False,
			collate_fn=lambda batch: variable_time_collate_fn(batch, args, device, data_type='test', data_min=data_min, data_max=data_max))

	##################################################################
	# Human activity dataset
	if dataset_name == 'activity':
		n_samples = min(10000, args.n)
		dataset_obj = PersonActivity(DATA_DIR / 'activity' if LOCAL else DATA_DIR, download=True, n_samples=n_samples, device=device)

		train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8, shuffle=False)
		train_data, val_data = model_selection.train_test_split(dataset_obj, train_size=0.75, shuffle=False)

		input_dim = train_data[0][2].shape[-1]
		output_dim = train_data[0][-1].shape[-1]

		batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
		dltrain = DataLoader(train_data, batch_size=batch_size, shuffle=False,
			collate_fn=lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type='train'))
		dlval = DataLoader(val_data, batch_size=batch_size, shuffle=False,
			collate_fn=lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type='test'))
		dltest = DataLoader(test_data, batch_size=batch_size, shuffle=False,
			collate_fn=lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type='test'))


	return input_dim, output_dim, dltrain, dlval, dltest
