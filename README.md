This is not an officially supported Google product. It is a set of scripts for a paper.

HISTORY Forked from https://github.com/lancifollia/tinygbt which originally stated MIT as its license

# Experimental Python Framwork for Accelerated Gradient Boosting Machine.
The tinygbt folder is adapted from the tinygbt repository by Seong-Jin Kim with
MIT license (https://github.com/lancifollia/tinygbt).

This repository details a Python framework which exposes an API to conduct
experiments on Accelerated Gradient Boosting Machine, and in particular the code
to reproduce the figures in paper: https://arxiv.org/pdf/1903.08708.pdf.

It has four scripts:

*   `cross_validation.py`: run different methods with cross validation on
    hyperparameters.
*   `solve_and_plot.py`: run different methods with fixed parameters and plot
    the figures.
*   `solve_libsvm_instances.py`: run different methods with fixed parameters.
*   `vagbm_solve.py`: compare the results between agbm and the method proposed
    in previous paper.

## Data preparation.
To prepare data for the experiments, download a1a, housing, w1a dataset
from LIBSVM Data at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/,
and save them at the data_directory. Provide this data directory via data_folder
flag to the scripts you run.

## Usage examples:
For run with cross_validation: `cross_validation.py --loss=LogisticLoss --dataset_name=housing --data_folder data_directory`

For run with solve_and_plot: `solve_and_plot.py --num_trees=100
--dataset_name=w1a --loss=LogisticLoss --min_split_gain=0.00001
--learning_rate=0.1 --data_folder=TODO --output_dir=TODO`

For run with vagbm_solve: `vagbm_solve.py  --num_trees=100
--dataset_name=housing --loss=L2Loss --min_split_gain=1e-10 --learning_rate=0.1
--data_folder=TODO --output_dir=TODO`

For tree stumps test: `solve_and_plot.py --num_trees=1000
--dataset_name=a1a --loss=LogisticLoss --min_split_gain=0.00001
--learning_rate=0.1 --max_depth=1 --data_folder data_directory`
