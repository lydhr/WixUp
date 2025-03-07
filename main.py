import os, sys
import random
import numpy as np
import hydra
from omegaconf import DictConfig
import logging
import optuna

import torch
from datasets import *

from utils import utils, constants, data_utils
from mmrnet.models import *
import mmrnet.session as session

if torch.cuda.is_available():
    # ensure CUDA deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" # will increase library footprint in GPU memory by approximately 24MiB
    torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.benchmark = False

utils.init_hydra()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg: DictConfig):
    # init
    if cfg.debug:
        sys.excepthook = utils._excepthook
    utils.set_seed(cfg.seed) # boilerplate: utils.set_seed(cfg.seed)
    constants.check_hydra_values(cfg)
    # run train, test, tune, selfTrain
    callable = globals().get(cfg.run)
    callable(cfg)

def selfTrain(cfg):
    # e.g. train = A-train, validate = B-test, test = B-train
    model, dataloaders = setup_model_and_data(cfg=cfg, partitions=['train', 'validate', 'test'])
    _, model = session.train(model=model, train_loader=dataloaders['train'], val_loader=dataloaders['validate'], cfg=cfg.model)

    x_yhat_test = session.test(model=model, test_loader=dataloaders['test'], cfg=cfg.model)
    x_yhat_test = [[x/100, y/100] for x, y in x_yhat_test] # ds transform Scale(100)

    #mix train&test-Pseudo as new train
    dataloaders['train'].dataset.mix_testData(x_yhat_test, cfg.augment)

    log.info(f"new train data size = {dataloaders['train'].dataset.len()}")

    #re-train and evaluate
    session.train(model=model, train_loader=dataloaders['train'], val_loader=dataloaders['validate'], cfg=cfg.model)

    return 

def train(cfg, trial_args=None):
    model, dataloaders = setup_model_and_data(cfg=cfg, trial_args=trial_args, 
                            partitions=['train', 'validate'])
    
    if trial_args:
        for key in ['learning_rate', 'weight_decay', 'optimizer']:
            if key in trial_args:
                cfg.model[key] = trial_args.get(key)

    loss = session.train(model=model, train_loader=dataloaders['train'], val_loader=dataloaders['validate'], cfg=cfg.model)
    return loss

def test(cfg):
    model, dataloaders = setup_model_and_data(cfg=cfg, partitions=['train', 'test'])

    session.test(model=model, test_loader=dataloaders['test'], cfg=cfg.model)
# cli_eval = cli_test

def tune(cfg):
    def objective(trial):
        # define hyperparameter search space
        # stack = trial.suggest_categorical("stack", [1, 3, 5, 25, 50, 75])
        # learning_rate = trial.suggest_categorical("learning_rate", [1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 1e-3, 1e-2])
        # weight_decay = trial.suggest_categorical("weight_decay", [0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5])
        stack = trial.suggest_categorical("stack", [1, 3, 5])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-6])
        weight_decay = trial.suggest_categorical("weight_decay", [0])

        val_loss = train(cfg,
            trial_args={'stacks': stack,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay})
        return val_loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.tuner.n_trials)
    print(study.best_trial)


def setup_model_and_data(cfg, partitions, trial_args=None):
    if trial_args:
        cfg.data.stacks = trial_args.get('stacks')
    # get dataset
    dataloaders, dataset_info = get_dataloaders(cfg=cfg, partitions=partitions)

    # get model
    model_cls = globals()[cfg.model.name]
    model = model_cls(info=dataset_info)
    return model, dataloaders

def get_dataloaders(cfg, partitions):
    # init datasets
    datasets = {}
    assert 'train' in partitions # for loading model info, e.g. num of classes
    for partition in partitions:
        dataset_class = globals()[cfg.data[partition]['set']]
        datasets[partition] = dataset_class(partition=partition, cfg=cfg.data)

    info = datasets['train'].info # num_classes matters, test set could be sub set
    for partition in partitions:
        _assert_classess_match(datasets[partition], datasets['train'])

    # init dataloaders
    dataloaders = {partition: data_utils.get_dataloader(datasets[partition], cfg.model, partition)
                    for partition in datasets}
    return dataloaders, info

def _assert_classess_match(ds1, ds2):
    cls1 = ds1.get_classes()
    cls2 = ds2.get_classes()
    if cls1 is None:
        assert cls2 == cls1
    else:
        assert all(cls2 == cls1)

if __name__ == "__main__":
    main()
