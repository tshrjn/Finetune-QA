from pprint import pprint

import wandb
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Custom Files
import opts
import data
import utils
import models

def experiment(args):
    utils.seed_everything(seed=args.seed)
    qa_model = models.QAModel(hparams=args)
    train_dl, valid_dl, test_dl = data.prepare_data(args)

    wandb_logger = WandbLogger(project='qa', entity='nlp', tags=args.tags, offline=args.fast_dev_run)
    wandb_logger.watch(qa_model, log='all')
    args.logger = wandb_logger

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(qa_model, train_dataloader=train_dl, val_dataloaders=valid_dl)
    trainer.test(qa_model, test_dataloaders=test_dl)



if __name__ == '__main__':
    args = opts.get_args()
    pprint(vars(args))
    experiment(args)