**Please note: this repository is no longer being maintained.**

# Additivity and Nonadditivity for ML in Drug Design

This repository contains code for paper: 

Karolina Kwapien, Eva Nittinger, Jiazhen He, Christian Margreitter, Alexey Voronov, and Christian Tyrchan. Implications of Additivity and Nonadditivity for Machine Learning and Deep Learning Models in Drug Design. ACS Omega 2022 7 (30), 26573-26581. DOI: [10.1021/acsomega.2c02738](https://doi.org/10.1021/acsomega.2c02738)

This repository contains code to run hyper-parameter optimization for RF, SVR, XGBoost, and PLS algorithms. The data is not included in this repository.

## Directories
  * root - Python and shell scripts for running hyper-parameter optimization.
  * `notebooks` - Jupyter Notebooks for splitting data and computing test scores.
  * `data-initial` - Initial data, not included in this repo.
  * `data` - Main data: random split of initial data into train and test data, not included in this repo.
  * `downsampled-10-percent` - Down-sampled 10% of main data, not included in this repo.
  * `optuna-storage` - Auxiliary storage for `optuna` library to track hyper-parameter optimization progress, not included in this repo.
  * `best-models` - Models with best hyper-parameters, not included in this repo.
  * `pred_values` - Predicted vs expected values for models with best hyper-parameters.
  * `fill-gaps-configs` - build configurations for best found hyper-parameters for "filling gaps" (see paper).

## Workflow

1. First, split initial data into training and test datasets using Jupyter Notebook.
2. Then run all 32 optimization jobs using script `submit_all_to_slurm_on_full_data.sh`.
3. If any of the jobs fails:
    * Prepare down-sampled data using Jupyter Notebook.
    * Re-submit failed optimization jobs using down-sampled data.
    * Prepare "fill-gaps" build configurations for the best-found hyper-parameters using Jupyter Notebook.
    * Submit "fill-gaps" build jobs.
4. Then prepare summary table using Jupyter Notebook.

## Dependencies

This code uses [QPTUNA](https://github.com/MolecularAI/qptuna) to set up hyper-parameter optimization.

Optimization jobs are started using SLURM, but they can be started without `SLURM` too.

## License

Apache 2.0.

## Contributors
- Christian Margreitter [@cmargreitter](https://github.com/CMargreitter)
- Alexey Voronov [@alexvoronov](https://github.com/AlexVoronov)
