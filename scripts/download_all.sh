#!/bin/sh

echo "GENERATING SYNTHETIC DATA"
python -m nfe.experiments.synthetic.generate


echo "PREPARING SMOOTHING EXPERIMENT DATA"
python -m nfe.experiments.latent_ode.mujoco_physics
python -m nfe.experiments.latent_ode.physionet
python -m nfe.experiments.latent_ode.person_activity


echo "GENERATING TEMPORAL POINT PROCESS SYNTHETIC DATA"
python -m nfe.experiments.tpp.generate


echo "PREPARING SPATIO-TEMPORAL DATA"
python -m nfe.experiments.stpp.data.download_and_preprocess_citibike
python -m nfe.experiments.stpp.data.download_and_preprocess_covid19
python -m nfe.experiments.stpp.data.download_and_preprocess_earthquakes
