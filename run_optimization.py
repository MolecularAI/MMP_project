import argparse
import logging
import os
import sys

from qptuna.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
)
from qptuna.config import ModelMode, OptimizationDirection
from qptuna.config.optconfig import (
    OptimizationConfig,
    PLS,
    RandomForest,
    SVR,
    XGBregressor,
)
from qptuna.datareader import Dataset
from qptuna.descriptors import ECFP


logger = logging.getLogger(__name__)


def main():
    algs = {
        "PLS": PLS.new(
            n_components={"low": 2, "high": 10}
        ), 
        "RF": RandomForest.new(
            max_features=["auto", "sqrt", "log2"],
            max_depth={"low": 2, "high": 32},
            n_estimators={"low": 10, "high": 250}
        ), 
        "SVR": SVR.new(), 
        "xgboost": XGBregressor.new(
            max_depth={"low": 2, "high": 32},
            n_estimators={"low": 10, "high": 300},
            learning_rate={"low": 0.1, "high": 0.1}  # Constant.
        )
    }

    props = ["Clearance", "logD", "Permeability", "Solubility"]

    sets = ["1", "2", "3", "4"]

    parser = argparse.ArgumentParser(description='Run MMP study.')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--prop", choices=props, required=True)
    requiredNamed.add_argument("--setid", choices=sets, required=True)
    requiredNamed.add_argument("--alg", choices=algs.keys(), required=True)
    requiredNamed.add_argument("--datadir", required=True)

    args = parser.parse_args()
    
    prop = args.prop
    setid = args.setid
    alg = args.alg
    datadir = args.datadir

    study_name = f"MMP_{datadir}_{prop}_set{setid}_{alg}"
        
    config = OptimizationConfig(
        data=Dataset(
            input_column="SMILES",
            response_column="VALUE",
            training_dataset_file=f"{datadir}/{prop}_set{setid}_train.csv",
        ),
        descriptors=[
            ECFP.new(radius=3, nBits=2048)  # For ECFP6, radius=3. 
        ],
        algorithms=[
            algs[alg],
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_jobs=8,
            cross_validation=3,
            n_trials=300,
            direction=OptimizationDirection.MAXIMIZATION,
            optuna_storage=f"sqlite:///optuna-storage/optuna_storage_{study_name}.sqlite",
            track_to_mlflow=False,
        ),
    )
    
    study = optimize(config, study_name=study_name)
    
    # Get the best Trial from the Study and make a Build (Training) configuration for it.
    buildconfig = buildconfig_best(study)
    
    # Build (re-Train) and save the best model.
    build_best(buildconfig, f"best-models/best-{study_name}.pkl")

    
if __name__ == '__main__':
    # Configure logger to output to console (Slurm collects console output by default).
    # For remote logging, we can add handlers HTTPHandler or SysLogHandler from logging.handlers.
    # For machine-readable logs, we can use a Json formatter, like https://github.com/madzak/python-json-logger.
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logging.captureWarnings(True)  # Capture sklearn warnings about version mismatch.

    exit(main())
