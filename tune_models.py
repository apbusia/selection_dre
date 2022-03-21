import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import run_models


SEED = 7


# Set up default hyperparameters.
hparam_defaults = {'model_type': 'ann',
                   'encoding': 'is',
                   'n_hidden': 3,
                   'hidden_size': 100,
                   'learning_rate': 1e-5,
                   'alpha': 0.,
                   'epochs': 10,
                   'batch_size': 128,
                   'early_stopping': True,
                   'normalize': False,
                   'weighted_loss': False,
                   'gradient_clip': None}

wandb.init(config=hparam_defaults)
config = wandb.config

data_df = pd.read_csv(config.data_file)
seqs = data_df['seq']
pre_counts = data_df[config.pre_column]
post_counts = data_df[config.post_column]

train_idx, val_idx = train_test_split(range(len(seqs)), test_size=0.2, random_state=SEED)
run_models.run_training(
    seqs, pre_counts, post_counts, config.encoding, config.model_type, config.normalize,
    lr=config.learning_rate, n_hidden=config.n_hidden, hidden_size=config.hidden_size, alpha=config.alpha,
    train_idx=train_idx, test_idx=val_idx, val_idx=val_idx, epochs=config.epochs, batch_size=config.batch_size,
    early_stopping=config.early_stopping, weighted_loss=config.weighted_loss, gradient_clip=config.gradient_clip,
    wandb=True, return_metrics=False)