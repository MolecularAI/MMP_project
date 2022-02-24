#!/bin/bash

mkdir -p optuna-storage
mkdir -p best-models
mkdir -p slurm-logs

for alg in PLS RF SVR xgboost; do
    for setid in 1 2 3 4; do
        for prop in Clearance logD Permeability Solubility; do
            echo $alg $prop $setid
            sbatch run_optimization_on_slurm.sh --prop=$prop --setid=$setid --alg=$alg --datadir=data
        done
    done
done
