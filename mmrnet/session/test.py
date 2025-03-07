import os
import torch
import pytorch_lightning as pl
import numpy as np
import logging


from .wrapper import ModelWrapper
from .visualize import make_video
from utils import utils

log = logging.getLogger(__name__)


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model

def test(model, test_loader, cfg):
    plt_model = ModelWrapper(model)
    if cfg.checkpoint.load_path is not None:
        checkpoint = utils.get_checkpoint_path(cfg.checkpoint)
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")

    plt_model.eval()
    plt_trainer_args = {
        'devices': cfg.device.num_devices,
        'accelerator': cfg.device.accelerator, 
        'strategy': cfg.device.strategy,
        'default_root_dir': cfg.checkpoint.save_path}
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, test_loader)

    if cfg.visualize:
        filename = checkpoint[:-5]+'.avi'
        log.info(f'Saving test result in {filename}')
        res = trainer.predict(plt_model, test_loader)
        Y_pred = np.concatenate([r[0] for r in res])
        Y = np.concatenate([r[1] for r in res])
        make_video(Y_pred, Y, filename)
        log.info(f'Saved {filename}')

    x_yhat = plt_model.x_yhat_test
    return x_yhat