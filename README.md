# Dec-QMIX

This is the codification used in the IROS 2025 paper, which examines the challenges of implementing Centralized Training and Decentralized Execution (CTDE) in a network of Unmanned Aerial Vehicles (UAVs) for monitoring dynamic wildfire fronts. 

The most prominent CTDE framework, QMIX has shown exceptional performance in cooperative multi-agent environments like the StarCraft 2 Multi-Agent Challenge (SMAC2). However, implementing CTDE in real-world multi-robot applications has significant limitations. The unpredictable and dynamic nature of scenarios such as wildfires makes training under CTDE frameworks impractical, due to difficulties in gathering sufficient high-fidelity data points and modeling such volatile environments for effective training. Moreover, the training must continue throughout the operation to learn and adapt to sparse and unpredictable events as they occur. 

To address these issues, we propose Decentralized QMIX (Dec-QMIX), a novel approach that bridges the gap between QMIX and Independent Q-Learning (IQL). Dec-QMIX learns a decentralized, factored joint action-value function, enabling adaptive coordination among UAVs. It employs a local per-agent mixing network for joint action-value estimation and a distributed consensus filter to achieve convergence to a unified joint action-value function in a decentralized manner.

<img src="figures/Dec-QMIX_arch.png">

## Installation Instructions
In this repository, we will see how we built our Dec-QMIX over the previous QMIX work. The original fine-tuned QMIX repository is [here](https://github.com/hijkzzz/pymarl2).

**Libraries needed before installation:**
- Git
- Anaconda 3 or Miniconda 3

**Step-by-step instructions**
Clone github page
```shell
git clone https://github.com/aralab-unr/Dec-QMIX.git
cd Dec-QMIX
```
Create new Conda environment and Install Python packages
```shell
conda create -n pymarl python=3.8 -y
conda activate pymarl

bash install_dependecies.sh
```
Set up StarCraft II Multi-Agent Challenge Environment:
```shell
bash install_sc2.sh
```

## Command Line Tools
To run an experiment:
**StarCraft II Multi-Agent Challenge Environment**
```shell
# Run StarCraft 2 Multi-Agent Environment with QMIX algorithm
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```
```shell
# Run StarCraft 2 Multi-Agent Environment with Dec-QMIX algorithm
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z run=default_gs runner=parallel_gs learner=qlearner_gs
```
Configure `env_args.map_name` to change the map e.g., `2c_vs_64zg`, `MMM2`, or any other environment.

**Wildfire Environment**
```shell
# Run Wildfire Environment with QMIX algorithm
python3 src/main.py --config=qmix --env-config=wildfire with run=default_gs2 runner=parallel_gs2 mac= n_mac_gs2 learner=qlearner_gs
```
```shell
# Run Wildfire Environment with Dec-QMIX algorithm
python3 src/main.py --config=qmix --env-config=wildfire with run=default_gs3 runner=parallel_gs3 mac= n_mac_gs2 learner=qlearner_gs
```

**Kill all training processes**
```shell
# all python and game processes of current user will quit.
bash clean.sh
```

## Contact
- [Gaurav Srikar](mailto:gauravsrikar@gmail.com)
- [Dr. Hung (Jim) La](mailto:hla@unr.edu)
