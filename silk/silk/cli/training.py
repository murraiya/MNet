# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from silk.config.core import instantiate_and_ensure_is_instance
from silk.config.model import load_model_from_checkpoint
from silk.logger import LOG
from torch.utils.data import DataLoader


def main(config: DictConfig):
    """Run a generic configurable training loop.

    Parameters
    ----------
    config : DictConfig
        Full configuraton structure. Important paths are described below.

    Paths
    ----------
    * `mode.model` : `LightningModule`

        Model to train.

    * `mode.loaders.training` : `DataLoader`

        Data loader used for training.

    * `mode.loaders.validation` : `DataLoader`

        Data loader used for validation.

    * `mode.trainer` : `Trainer`

        Trainer engine to run.

    Returns
    -------
    [type]
        [description]

    TODO(Pierre) : See TODO at end of function.
    """
    model = instantiate_and_ensure_is_instance(config.mode.model, pl.LightningModule)
    # here occurs silk.models.silk _init_loss_flow() 

    if config.mode.continue_from_checkpoint is not None:
        LOG.warning(
            f"the model's weight are being loaded from checkpoint : {config.mode.continue_from_checkpoint}\nplease make sure to disable it if that's not the intended behavior (by setting `config.mode.continue_from_checkpoint` to `null`)."
        )
        load_model_from_checkpoint(
            model,
            checkpoint_path=config.mode.continue_from_checkpoint,
            strict=True,
            freeze=False,
            eval=False,
        )
    # print(config.mode.loaders.training)
    train_loader = instantiate_and_ensure_is_instance(
        config.mode.loaders.training, DataLoader
    )
    val_loader = instantiate_and_ensure_is_instance(
        config.mode.loaders.validation, DataLoader
    )
    print(len(train_loader)) #42728
    print(len(val_loader)) #2114
    print(type(train_loader))
    # for batch in train_loader:
    #     # img = batch
    #     print(batch.size(), type(batch))
    
    trainer = instantiate_and_ensure_is_instance(config.mode.trainer, pl.Trainer)
    # trainer = pl.Trainer(
    #     val_check_interval=0.25,
    #     strategy= "ddp",
    #     gpus = 1,
    #     benchmark= True,
    #     max_epochs= 100,
    #     limit_val_batches= 100,
    #     limit_train_batches= 1000
    # )
    # TEMPORARY(Pierre) : disabled because of missing transform and loss.
    # print("!!!!!!!!!!!!!!!!!!!!", model)
     
    trainer.fit(model, train_loader, val_loader) #usually this becomes train
    
    # TEMPORARY(Pierre) : return some value for quick testing of pipeline
    # TODO(Pierre) : remove and replace by loss value ?
    return {
        "model": model,
        "val_loader": val_loader,
        "train_loader": train_loader,
        "trainer": trainer,
        "config": OmegaConf.to_object(config),
    }
