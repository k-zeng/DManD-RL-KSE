# DManD-RL-KSE
Demo Code for DManD-RL Applied to the Kuramoto-Sivashinsky Equation

Run main.py to generate demo trajectory: generates from specified initial condition (IC#) an RL agent controlled trajectory in the true domain (semi-implicit KSE solver) and the corresponding RL agent controlled trajectory in the NODE--decoded. The statistical properties of the trajectories are also provided.

You will need the following packages:
tensorflow: 1.14.0
pytorch: 1.9.0
torchdiffeq: 0.2.2
scipy: 1.7.3
scikit-learn: 1.0.2
h5py: 2.10.0
seaborn: 0.12.0
