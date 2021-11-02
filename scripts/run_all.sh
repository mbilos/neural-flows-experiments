#!/bin/sh

### Synthetic
# Ellipse
python -m nfe.train --seed 1 --experiment synthetic --data ellipse --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model ode --odenet concat --solver rk4 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment synthetic --data ellipse --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model coupling --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment synthetic --data ellipse --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model resnet --flow-layers 4 --time-net TimeFourierBounded --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
# Sawtooth
python -m nfe.train --seed 1 --experiment synthetic --data sawtooth --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model ode --odenet concat --solver rk4 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment synthetic --data sawtooth --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model coupling --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment synthetic --data sawtooth --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model resnet --flow-layers 4 --time-net TimeFourierBounded --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
# Sink
python -m nfe.train --seed 1 --experiment synthetic --data sink --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model ode --odenet concat --solver rk4 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment synthetic --data sink --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model coupling --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment synthetic --data sink --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model resnet --flow-layers 4 --time-net TimeFourierBounded --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
# Square
python -m nfe.train --seed 1 --experiment synthetic --data square --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model ode --odenet concat --solver rk4 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment synthetic --data square --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model coupling --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment synthetic --data square --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model resnet --flow-layers 4 --time-net TimeFourierBounded --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
# Triangle
python -m nfe.train --seed 1 --experiment synthetic --data triangle --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model ode --odenet concat --solver rk4 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment synthetic --data triangle --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model coupling --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment synthetic --data triangle --epochs 1000 --batch-size 50 --weight-decay 1e-05 --model flow --flow-model resnet --flow-layers 4 --time-net TimeFourierBounded --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity


### Smoothing
# Activity
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data activity --hidden-layers 3 --hidden-dim 100 --rec-dims 30 --latents 20 --gru-units 100 --model ode --odenet concat --solver euler
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data activity --hidden-layers 3 --hidden-dim 100 --rec-dims 30 --latents 20 --gru-units 100 --model ode --odenet concat --solver dopri5
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data activity --hidden-layers 3 --hidden-dim 100 --rec-dims 30 --latents 20 --gru-units 100 --model flow --flow-model coupling --flow-layers 2 --time-net TimeLinear --time-hidden-dim 8
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data activity --hidden-layers 3 --hidden-dim 100 --rec-dims 30 --latents 20 --gru-units 100 --model flow --flow-model resnet --flow-layers 2 --time-net TimeTanh --time-hidden-dim 8
# MuJoCo / Hopper
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 0 --data hopper --hidden-layers 3 --hidden-dim 100 --rec-dims 100 --latents 20 --gru-units 50 --model ode --odenet concat --solver euler
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 0 --data hopper --hidden-layers 3 --hidden-dim 100 --rec-dims 100 --latents 20 --gru-units 50 --model ode --odenet concat --solver dopri5
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 0 --data hopper --hidden-layers 2 --hidden-dim 100 --rec-dims 100 --latents 20 --gru-units 50 --model flow --flow-model coupling --flow-layers 2 --time-net TimeLinear --time-hidden-dim 8
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 0 --data hopper --hidden-layers 2 --hidden-dim 100 --rec-dims 100 --latents 15 --gru-units 50 --model flow --flow-model resnet --flow-layers 2 --time-net TimeTanh --time-hidden-dim 8
# Physionet
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data physionet --hidden-layers 3 --hidden-dim 50 --rec-dims 40 --latents 20 --gru-units 50 --model ode --odenet concat --solver euler
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data physionet --hidden-layers 3 --hidden-dim 50 --rec-dims 40 --latents 20 --gru-units 50 --model ode --odenet concat --solver dopri5
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data physionet --hidden-layers 3 --hidden-dim 50 --rec-dims 40 --latents 20 --gru-units 50 --model flow --flow-model coupling --flow-layers 2 --time-net TimeLinear --time-hidden-dim 8
python -m nfe.train --seed 1 --experiment latent_ode --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --data physionet --hidden-layers 3 --hidden-dim 50 --rec-dims 40 --latents 20 --gru-units 50 --model flow --flow-model resnet --flow-layers 2 --time-net TimeTanh --time-hidden-dim 8


### Filtering
# MIMIC-III
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic3 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model ode --odenet gru --solver euler --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic3 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model ode --odenet gru --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic3 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model flow --flow-model gru --flow-layers 4 --time-net TimeTanh --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic3 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model flow --flow-model resnet --flow-layers 4 --time-net TimeTanh --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
# MIMIC-IV
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic4 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model ode --odenet gru --solver euler --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic4 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model ode --odenet gru --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic4 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model flow --flow-model gru --flow-layers 4 --time-net TimeTanh --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity
python -m nfe.train --seed 1 --experiment gru_ode_bayes --data mimic4 --epochs 1000 --batch-size 100 --weight-decay 0.0001 --lr-decay 0.33 --lr-scheduler-step 20 --model flow --flow-model resnet --flow-layers 4 --time-net TimeTanh --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity


### TPP (only real-world data & no marks)
# Mooc
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver euler --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder mixture --rnn gru
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder mixture --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --flow-model coupling --flow-layers 1 --time-net TimeLinear --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --flow-model resnet --flow-layers 1 --time-net TimeLinear --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model coupling --time-net TimeLinear --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model resnet --time-net TimeTanh --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model resnet --time-net TimeTanh --rnn gru
python -m nfe.train --seed 1 --experiment tpp --data mooc --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model rnn --hidden-dim 64
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver euler --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder continuous
# Reddit
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder mixture --rnn gru
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder mixture --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --flow-model coupling --flow-layers 1 --time-net TimeLinear --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --flow-model resnet --flow-layers 1 --time-net TimeLinear --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model coupling --time-net TimeLinear --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model resnet --time-net TimeTanh --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model resnet --time-net TimeTanh --rnn gru
python -m nfe.train --seed 1 --experiment tpp --data reddit --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model rnn --hidden-dim 64
# Wiki
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver euler --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder mixture --rnn gru
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model ode --odenet concat --solver dopri5 --hidden-layers 3 --hidden-dim 64 --activation ELU --final-activation Tanh --decoder mixture --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --flow-model coupling --flow-layers 1 --time-net TimeLinear --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --flow-model resnet --flow-layers 1 --time-net TimeLinear --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --decoder continuous
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model coupling --time-net TimeLinear --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model resnet --time-net TimeTanh --rnn lstm
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model flow --decoder mixture --flow-layers 1 --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --activation ReLU --final-activation Identity --flow-model resnet --time-net TimeTanh --rnn gru
python -m nfe.train --seed 1 --experiment tpp --data wiki --epochs 1000 --batch-size 50 --weight-decay 0.0001 --model rnn --hidden-dim 64


### Neural STPP
# Bike
python -m nfe.train --seed 1 --experiment stpp --data bike --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model ode --density-model independent --hidden-layers 4 --hidden-dim 64
python -m nfe.train --seed 1 --experiment stpp --data bike --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model ode --density-model attention --hidden-layers 4 --hidden-dim 64
python -m nfe.train --seed 1 --experiment stpp --data bike --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model flow --density-model independent --flow-layers 16 --hidden-dim 64 --hidden-layers 4 --time-net TimeLinear
python -m nfe.train --seed 1 --experiment stpp --data bike --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model flow --density-model attention --flow-layers 16 --hidden-dim 64 --hidden-layers 4 --time-net TimeLinear
# Covid
python -m nfe.train --seed 1 --experiment stpp --data covid --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model ode --density-model independent --hidden-layers 4 --hidden-dim 64
python -m nfe.train --seed 1 --experiment stpp --data covid --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model ode --density-model attention --hidden-layers 4 --hidden-dim 64
python -m nfe.train --seed 1 --experiment stpp --data covid --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model flow --density-model independent --flow-layers 16 --hidden-dim 64 --hidden-layers 4 --time-net TimeLinear
python -m nfe.train --seed 1 --experiment stpp --data covid --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model flow --density-model attention --flow-layers 16 --hidden-dim 64 --hidden-layers 4 --time-net TimeLinear
# Earthquake
python -m nfe.train --seed 1 --experiment stpp --data earthquake --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model ode --density-model independent --hidden-layers 4 --hidden-dim 64
python -m nfe.train --seed 1 --experiment stpp --data earthquake --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model ode --density-model attention --hidden-layers 4 --hidden-dim 64
python -m nfe.train --seed 1 --experiment stpp --data earthquake --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model flow --density-model independent --flow-layers 16 --hidden-dim 64 --hidden-layers 4 --time-net TimeLinear
python -m nfe.train --seed 1 --experiment stpp --data earthquake --epochs 1000 --batch-size 50 --weight-decay 0.0001 --patience 50 --model flow --density-model attention --flow-layers 16 --hidden-dim 64 --hidden-layers 4 --time-net TimeLinear
