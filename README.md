<div align="center">

# 3D CT Forams: foraminifera detection in 3D volumes

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

**This work is carried out as part of a PNRIA project.**

[Geoffroy Couasnet](), [Romain Karpinski](mailto:romain.karpinski@loria.fr), [Aurelia Mouret](), [Maria Pia Nardelli](), [Nicolas Vanderesse]()

</div>

# Installation

## Getting Started

`conda create --name ctforams python=3.10`

`pip install -r requirements.txt`

## Train a model

```sh
HYDRA_FULL_ERROR=1 python3 ctforams/train.py\
    task_name="aq1"\ # Task name
    data.batch_size=32\ # Batch size
    hdf5_name=Aq1Aq3.h5\ # HDF5 file containing the training data (<code base>/data/<hdf5_name>)
    data.num_workers=19\ # Number of parallel workers (number optimized for JZ)
```

## Evaluate a model

```sh
HYDRA_FULL_ERROR=1 python3 ctforams/eval.py\
    model_dir=${model_dir}\ # The path to the model dir
    +data.batch_size=16\ # Batch size
    progress=True\ # Show progress during inference
    +hdf5_name=${hdf5_name}\ # The hdf5 path
    +data.test_data.group_name=test\ # You can override the key used for the test (train/val/test)
    threshold=0.8\ # Prob map threshold
```

# Jean-Zay installation

This section describes how to run an experiment on Jean-Zay HPC

## Environment setup

- http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-python-env-eng.html
- http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-calculateurs-disques-eng.html

## Installation steps

1. Follow previous section to create a personnal python env
2. Clone the repository into `$WORK`: `cd $WORK` then `git clone https://github.com/N-Van/3D_CT-Forams`
3. Install conda env:

   1. conda create -n ctforams python=3.10
   2. conda activate ctforams
   3. cd $WORK/3D_CT-Forams
   4. pip install -r requirements.txt
   5. Créer un lien symbolique $ALL_CCFRWORK/data -> $WORK/data

4. Test de l'installation via une réservation intéractive:
   Récupérer le script run_segmentation.sh et le rendre exécutable chmod +x run_segmentation.sh
   srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -A cpp@v100 --time=01:00:00 --qos=qos_gpu-dev bash

5. From the node:
   1. Activate conda env `conda activate ctforams`
   2. Run the following script (later called `run_segmentation.sh`)

```sh
# Move to the source code
cd $WORK/3D_CT-Forams
HYDRA_FULL_ERROR=1 python3 ctforams/train.py\
    task_name="test_run"\ # Task name
    data.batch_size=32\ # Batch size
    hdf5_name=crops/hdf5/loo/all_but_Aq1T0_270z_770z_2.h5\ # HDF5 file containing the training data (<code base>/data/<hdf5_name>)
    data.num_workers=19\ # Number of parallel workers (number optimized for JZ)
```

## Using a slurm configuration file

Here is a job reservation config file for slurm called `segmentation.slurm`
It can be used: `sbatch segmentation.slurm <command>`

```sh
#!/bin/bash
#SBATCH --job-name=segmentation      # job name
#SBATCH -C v100-32g                  # Use a V100 with 32Go VRAM
#SBATCH -A cpp@v100                  # which projet to use
#SBATCH --ntasks=1                   # one task
#SBATCH --ntasks-per-node=1          # one task per node
#SBATCH --gres=gpu:1                 # one gpu per node
#SBATCH --cpus-per-task=10           # 10 cpu is 1/8 of the whole cpu node of 8 gpus
#SBATCH --hint=nomultithread         # hyperthreading disabled
#SBATCH --time=02:00:00              # max execution time (HH:MM:SS)
#SBATCH --output=pyseg%j.out      # output file name
#SBATCH --error=pyseg%j.out       # error file name

# Clean all loaded module
module purge

# Load env var to find conda for instance
source $HOME/.bashrc
# Activate our conda env
conda activate ctforams

# Write all executed commands
set -x

# We run the script that is given in parameter of this file
srun $@
```

This script is really generic. We define the reservation parameters then we forward the command to run (i.e `./run_segmentation.sh` script)

All data ressources are located here: `$ALL_CCFRWORK/data`
**A data backup exists in `$ALL_CCFRSTORE`**

## Evaluation script

Usage: `sbatch segmentation.slurm ./eval.sh

```sh
#!/bin/bash
set -x

# Move
cd $WORK/3D_CT-Forams

# Model directory relative to the current folder
# 3D_CT-Forams/logs/Aq1T3/runs/2025-03-13-16-14/
# Must be logs/Aq1T3/runs/2025-03-13-16-14/
model_dir=$1

# The hdf5 containing the data with either train/val/test data
# Relative to the <current_folder>/data
# 3D_CT-Forams/data/crops/hdf5/test/Aq1T2.h5
# Must be crops/hdf5/test/Aq1T2.h5
hdf5_name=$2

export TMPDIR=$WORK/tmp

HYDRA_FULL_ERROR=1 python3 ctforams/eval.py\
        model_dir=${model_dir}\ # The path to the model dir
        +data.batch_size=16\ # Batch size
        progress=True\ # Show progress during inference
        +hdf5_name=${hdf5_name}\ # The hdf5 path
        +data.test_data.group_name=test\ # You can override the key used for the test (train/val/test)
        threshold=0.8\ # Prob map threshold
```

## A few useful slurm commands:

`squeue -u $USER`: show all running and pending jobs (with ids)
`scancel <jobid>`: cancel a job using its id
`scancel -u $USER`: cancel all jobs
`scontrol show job <jobid>`: show details on a job. we can see here when it is scheduled to start.
`srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -A cpp@v100 --time=01:00:00 --qos=qos_gpu-dev bash`: interactive reservation (useful for a first run or to debug). Here we ask for one hour using the development qos ([see here for more information on QoS](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html))
`srun --jobid=<JOBID> --pty /usr/bin/bash`: allows get a shell to an existing running job (to look for memory usage, gpu usage...)

- Other ressources:
  - `idracct`: show hours used and left on a project on JZ
  - Tmux/screen

## Volume prediction

```

```
