# DManD-RL-KSE
Demo Code for DManD-RL Applied to the Kuramoto-Sivashinsky Equation

Run main.py to generate a demo RL controlled trajectory: generates from specified initial condition (IC#) an RL agent controlled trajectory in the true domain (semi-implicit KSE solver) and the corresponding RL agent controlled trajectory in the NODE--decoded. The statistical properties of the trajectories are also provided. The training data for PCA, autoencoders, and NODE are located in the file Data.p.

You will need the following packages: <br />
python: 3.7.11 <br />
tensorflow: 1.14.0 <br />
pytorch: 1.9.0 <br />
torchdiffeq: 0.2.2 <br />
scipy: 1.7.3 <br />
scikit-learn: 1.0.2 <br />
h5py: 2.10.0 <br />
seaborn: 0.12.0 <br />
