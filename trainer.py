import os
import json
from typing import Any, Callable, Dict, Optional, Tuple, Union

from matplotlib import pyplot as plt
import torch
from utils import data_loader, losses, models, traintest

import pandas as pd
import numpy as np
import torch.nn as nn
import datetime

class Trainer(object):

    cutoff_date = datetime.datetime.strptime('2004-01-01', '%Y-%m-%d')
    end_date = '2018-06-30'

    y_names = ('Eq', 'FI', 'Macr', 'Cmdty', 'FX')

    def __init__(self,
                 classification : bool = True,
                 encoder_class : Optional[type] = None,
                 encoder_params : Dict[str, Any] = {},
                 decoder_class : Optional[type] = None,
                 decoder_params : Dict[str, Any] = {},
                 map_layer_dims : Tuple[int] = (10,),
                 dropout : float = 0.1,
                 activations : Dict[str, nn.Module] = dict(initial = nn.Tanh(),
                                                           H = nn.Tanh(),
                                                           loss = nn.Sigmoid(),
                                                           final = nn.Identity()),
                 sequential : bool = False,
                 seq_len : Optional[int] = 21,
                 *args, **kwargs):
        
        self.classification = classification
        self.sequential = sequential
        self.seq_len = seq_len

        self.pd_X, self.pd_y = self.load_dataframes()
        self.X_names_in = self.pd_X.columns
        self.Y_names_in = self.pd_y.columns
        self.Y_name_types = {y: float for y in self.Y_names_in}

        self.input_dim = len(self.X_names_in)
        self.representation_dim = len(self.Y_names_in)


        encoder = encoder_class(**encoder_params,
                                classification = classification,
                                activations = activations) if encoder_class is not None else None
        self.saemodel = models.encoder_decoder(X_dim_in = self.input_dim,
                                               map_layer_dims = list(map_layer_dims),
                                               encoding_layer_dim = self.representation_dim,
                                               dropout_p = dropout,
                                               classification = classification,
                                               external_enc = encoder,
                                               activations = activations)

    def train(self,
              num_epochs : int = 40,
              batch_size : int = 50,
              lr : float = 0.005,
              lr_scheduler : Optional[Dict[str, Union[type, Dict[str, Any]]]] = None,
              weight_decay : float = 1e-5,
              loss : Callable = losses.loss_fixedW,
              sup_loss_scale : float = 0.005,
              *args, **kwargs):
        
        train_loader, test_loader, _, _ = data_loader.load_sent_data(datasetclass = data_loader.sent_Dataset_sequence if self.sequential else data_loader.sent_Dataset,
                                                                   seq_len = self.seq_len,
                                                                   train_batch_size = batch_size,
                                                                   test_batch_size = batch_size,
                                                                   df = self.pd_X.join(self.pd_y),
                                                                   df_cutoff = self.cutoff_date,
                                                                   Y_names = self.Y_names_in,
                                                                   Y_name_types = self.Y_name_types,
                                                                   X_names = self.X_names_in,
                                                                   train_dates = pd.date_range(start=self.cutoff_date, end=self.end_date),
                                                                   convert_Y_to_labels = False,
                                                                   shuffle = True)
        self.test_loader = test_loader
        
        supervised_loss = nn.BCELoss if self.classification else nn.MSELoss
        sub_losses = (nn.MSELoss(),) + (supervised_loss(),) * self.representation_dim
        self.sae = traintest.sae(model_in = self.saemodel,
                                 loss_fct_in = loss,
                                 sub_losses_in = list(sub_losses),
                                 train_loss_wgt = False,
                                 lr_in = lr,
                                 scheduler_in = lr_scheduler,
                                 sub_losses_in_wgt_init = (10 / self.representation_dim,) + (10 * sup_loss_scale / self.representation_dim,) * self.representation_dim,
                                 classification = self.classification,
                                 firsttrain_reconstruction = 10,
                                 l2strength = weight_decay,
                                 sparse_regularizer= 0.0)
        self.sae.train(num_epochs, train_loader, test_loader)
        return np.array(self.sae.all_losses), np.array(self.sae.all_losses_val)

    @torch.no_grad()
    def predict(self):
        self.saemodel.eval()
        y_hat, y = [], []
        for xs, ys, _ in self.test_loader:
            y_hat.append(self.saemodel.enc(xs.to(self.sae.device))[0])
            y.append(ys)

        y_hat = torch.concat(y_hat).squeeze().detach().cpu().numpy()
        y = torch.concat(y).squeeze().detach().cpu().numpy()

        th = 0.5 if self.classification else 0
        y_hat_disc = y_hat > th
        y_disc = y > 0

        accs = np.mean(y_hat_disc == y_disc, axis = 0)
        corrs = np.array([np.corrcoef(y[:, i], y_hat[:, i])[0, 1] for i in range(self.representation_dim)])
        return accs, corrs
    
    @staticmethod
    def load_dataframes():
        # sample data (with some random noise overlay)
        pd_X = pd.read_csv('https://www.dropbox.com/scl/fi/7moh3dehhxxm3qp0x05c4/sample_data_X.csv?rlkey=8f7su6vcj9c22gqui5oj13wdm&dl=1',index_col='DATE' )
        pd_y = pd.read_csv('https://www.dropbox.com/scl/fi/4jjzh0zhz9jo950w7h250/sample_data_y.csv?rlkey=y9v5ntkdbvunn04g8t43h62yi&dl=1',index_col='DATE' )
        pd_X.index = pd.to_datetime(pd_X.index,format='%Y-%m-%d')
        for c in pd_X.columns:
            pd_X[c] = pd_X[c].astype(np.float32)
        pd_y.index = pd.to_datetime(pd_y.index,format='%Y-%m-%d')
        return pd_X, pd_y
    
def compare_runs(*hyperparam_dicts : Dict[str, Union[str, Dict[str, Any]]],
                 exp_name : str,
                 save_dest : str = 'experiments',
                 ):

    exp_dir = os.path.join(save_dest, exp_name)
    os.makedirs(exp_dir)
    
    results = {}

    metric_col_names = ['reconstruction loss'] + [f'supervision loss ({feat_name})' for feat_name in Trainer.y_names] + ['total loss'] + [f'accuracy ({feat_name})' for feat_name in Trainer.y_names]

    logs_dir = os.path.join(exp_dir, 'Logs')
    os.mkdir(logs_dir)
    
    for i, hyperparams in enumerate(hyperparam_dicts):
        config_name = hyperparams.get('name', f'training {i+1}')
        
        model_params = hyperparams.get('model', {})
        training_params = hyperparams.get('training', {})

        trainer = Trainer(**model_params)
        train_summ, val_summ = trainer.train(**training_params)
        accs, corrs = trainer.predict()

        config_fname = config_name.replace(' ', '_')
        colnames = [f'train {c}' for c in metric_col_names] + [f'validation {c}' for c in metric_col_names]
        summaries = np.concatenate([train_summ, val_summ], axis = -1)
        pd.DataFrame(summaries, columns = colnames).to_csv(os.path.join(logs_dir, f'{config_fname}_logs.csv'), index = False)

        results[config_name] = dict(config = hyperparams,
                                    train_metircs = train_summ,
                                    val_metrics = val_summ,
                                    accuracies = accs,
                                    correlations = corrs)
    
    test_results = {name: {**{f'accuracy ({feat_name})': acc for feat_name, acc in zip(Trainer.y_names, result_dict['accuracies'])},
                           **{f'correlation ({feat_name})': acc for feat_name, acc in zip(Trainer.y_names, result_dict['correlations'])}}
                           for name, result_dict in results.items()
                    }
    results_df = pd.DataFrame(test_results).T
    results_df.to_csv(os.path.join(exp_dir, 'results.csv'))

    figs_dir = os.path.join(exp_dir, 'Figures')
    os.mkdir(figs_dir)

    for i, metric_col_name in enumerate(metric_col_names):
        plt.figure()
        for name, result_dict in results.items():
            values = result_dict['val_metrics'][:, i]
            plt.plot(values, label = name)
        
        plt.xlabel('epoch')
        plt.ylabel(metric_col_name)
        
        plt.grid()
        plt.legend()

        metric_col_fname = metric_col_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(figs_dir, f'{metric_col_fname}_plot.pdf'), bbox_inches = 'tight')
        