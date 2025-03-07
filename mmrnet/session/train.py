from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from .wrapper import ModelWrapper
import os
import torch

def train(
        model, train_loader, val_loader,
        cfg
        ):
    plt_model = ModelWrapper(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        optimizer=cfg.optimizer)
    metric = f'val_{plt_model.metric_name}'
    if 'mle' in metric:
        mode = 'min'
    elif 'acc' in metric:
        mode = 'max'
    else:
        raise ValueError(f'Unknown metric {metric}')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric,
        mode=mode,
        filename="best",
        dirpath=cfg.checkpoint.save_path,
        save_last=True,
    )

    if cfg.checkpoint.resume_last:
        checkpoint = os.path.join(cfg.checkpoint.save_path, 'last.ckpt')
        state_dict = torch.load(checkpoint)['state_dict']
        model.load_state_dict(state_dict)
        log.info(f'Loaded checkpoint from {checkpoint}')

    plt_model.train()
    plt_trainer_args = {
        'max_epochs': cfg.epochs,
        'devices': cfg.device.num_devices,
        'accelerator': cfg.device.accelerator,
        'strategy': cfg.device.strategy,
        'fast_dev_run': cfg.debug,
        'callbacks': [checkpoint_callback],
        'default_root_dir': cfg.checkpoint.save_path}
    #plt_trainer_args['enable_progress_bar'] = False
    trainer = pl.Trainer(**plt_trainer_args)
        
    trainer.fit(plt_model, train_loader, val_loader)
    return plt_model.best_val_loss, plt_model.model
