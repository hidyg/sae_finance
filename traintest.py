import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from losses import sparse_loss
# tensorboard
# import tensorflow as tf
from typing import Callable, List, Tuple




class sae:
    def __init__(self,
                 model_in: nn.Module,
                 loss_fct_in: Callable[ [torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,
                                         List[torch.Tensor],List[torch.Tensor] ], Tuple[ torch.Tensor, List[torch.Tensor]]  ],
                 sub_losses_in: List[nn.Module] = [ nn.CrossEntropyLoss() , nn.L1Loss() ],
                 sub_losses_in_wgt_init: List[float] = [ np.sqrt(0.1),  np.sqrt(20.0)],
                 lr_in: float = 0.01, # eta
                 optim_in: optim.Optimizer = optim.Adam,
                 l2strength: float = 0.0,
                 l1strength: float = 0.0,
                 train_loss_wgt: bool = True,
                 classification: bool = False,
                 firsttrain_reconstruction: int = 0,
                 sparse_regularizer: float = 0.0
                 ):
        '''
        wrapper class handling training, storing of trained models
        :param model_in: model object input (based on model class in models_ae)
        :param loss_fct_in: loss function input (out of losses in losses_ae)
        :param sub_losses_in: sub losses as list (mostly from nn.****)
        :param sub_losses_in_wgt_init: weights of losses (initial weights if trained, fixed if not trained)
        :param lr_in: learning rate
        :param optim_in: the optimizer ... Adam, pure SGD+ nesterovmom
        :param l2strength: L2 reg on weights
        :param train_loss_wgt:  train loss weights along the way
        :param classification: switch on the type of additional losses ( MSE possible. currently: should be True, want to do classification)
        '''
        self.model = model_in
        self.loss_fct = loss_fct_in
        self.lr_in = lr_in
        self.sub_losses = sub_losses_in
        self.sub_losses_wgt = sub_losses_in_wgt_init
        self.optim_in = optim_in
        self.l2_strength = l2strength
        self.l1_strength = l1strength
        self.classification = classification
        self.train_loss_wgt = train_loss_wgt
        self.all_losses = np.zeros(len(sub_losses_in_wgt_init))
        self.firsttrain_reconstruction = firsttrain_reconstruction
        self.sparse_regularizer = sparse_regularizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # to do: include validation set ...
    # evaluate on val (per epoch)
    def train(self,
              epochs: int,
              train_loader: torch.utils.data.Dataset,
              val_loader: torch.utils.data.Dataset = None
              ) -> None:
        '''
        :param epochs: passes over the data
        :param train_loader:  needs a torch data loader
        :param val_loader:  needs a torch data loader
        :return: None
        '''

        self.model.to(torch.device( self.device ))

        # epochs = 20
        # classificationLoss = nn.NLLLoss() # torch.nn.NLLLoss().cuda()   #
        # reconstructionLoss = nn.L1Loss() # nn.L1Loss().cuda()  # Potts: nn.MSEloss()
        # losses = [classificationLoss, reconstructionLoss]

        lossweights = [torch.tensor(w, dtype=torch.float32, requires_grad=self.train_loss_wgt) for w in self.sub_losses_wgt ]

        # in pytorch: tensor and variable "merged"-> need only tensor with grad = True
        # myvar = torch.tensor(0, dtype=torch.float32, requires_grad=True)

        # collect all parameters
        # get all parameters (model parameters + task dependent log variances)
        if self.train_loss_wgt:  # train the weights?
            params = ([p for p in self.model.parameters()] + lossweights)
        else:
            params = ([p for p in self.model.parameters()] )
            #
            # zero out classification / MSE losses for first x epochs
            sub_losses_wgt_first = [self.sub_losses_wgt[0]] + [0.0] * (len(self.sub_losses_wgt) - 1)
            lossweights_first_epochs = [torch.tensor(w, dtype=torch.float32, requires_grad=self.train_loss_wgt) for w in
                                        sub_losses_wgt_first]

        optimizer = self.optim_in(params,
                                  lr=self.lr_in, weight_decay=self.l2_strength )  # ADAM -> adaptive learning rate
        # optimizer = optim.SGD(DNN.parameters(), lr=0.001, momentum=0.8)  # ADAM -> adaptive learning rate

        batchidx = 0
        all_epoch_losses = []
        all_epoch_losses_val = []

        for epoch in range(epochs):
            self.model.train()
            epch_loss = 0.0  # per epoch loss
            epch_acc = [0.0] *(len(self.sub_losses)-1)

            component_losses = [0.0 for x in self.sub_losses ]

            epch_loss_val = 0.0
            component_losses_val = [0.0 for x in self.sub_losses]

            # for batch_idx, (data, label) in enumerate(train_loader):

            # in the firsttrain_reconstruction epochs, only train the reconstruction loss
            if not self.train_loss_wgt and epoch<self.firsttrain_reconstruction and self.classification:
                losswgt_in = lossweights_first_epochs
            else:
                losswgt_in = lossweights

            for (X, y, idxs) in train_loader:
                batchidx += 1
                #X = torch.Tensor(X).to(self.device)
                #y = torch.Tensor(y).to(self.device)
                X = X.to(self.device)
                y = y.to(self.device)

                # squeeze necessary (altern...change trainloader (to do))
                if len(y.shape) > 2:
                    y = y.squeeze(1)

                optimizer.zero_grad()
                y_hat, x_hat, _, activations_out = self.model(X)
                y_hat = y_hat
                x_hat = x_hat
                loss, loss_components = self.loss_fct(y_hat, x_hat, y, X, losswgt_in, self.sub_losses)

                # add sparsity loss
                loss += self.sparse_regularizer * sparse_loss( activations_out )
                loss.backward()
                optimizer.step()

                # add up mini batch losses
                epch_loss += loss.item() / len(train_loader)

                # get accuracies
                if self.classification:
                    for i in range(0,(len(self.sub_losses)-1)):
                        # only first classification at this stage
                        _, pred_class = torch.max(y_hat[:,(i*2):(i*2+2)].data, 1)  # returns tuple
                        # add up mini batch acc
                        epch_acc[i] += (pred_class.unsqueeze(1) == y[:,i].unsqueeze(1) ).sum().item() / (len(train_loader) * X.size(0))

                # loss components
                for k in range(0, len(self.sub_losses)):
                    component_losses[k] += loss_components[k].item() / len(train_loader)

            all_epoch_losses.append( component_losses + [epch_loss] + epch_acc )
            print('SAE train: Batch Nr', batchidx, ' iteration')
            print('SAE train: Epoch: ', epoch, ' - train_loss: ', epch_loss)
            print('SAE train: Epoch: ', epoch, ' - train_accuracies  (if classif.): ', epch_acc)
            printout = 'SAE train: Epoch: '+ str( epoch )
            printout = printout + '. '.join([' - loss_' +str(lnr)+':' + str(component_losses[lnr])   for lnr in range(0,len(self.sub_losses))])
            print(printout) #'SAE train: Epoch: ', epoch, ' - loss_1: ', component_losses[0], ' - loss_2: ', component_losses[1]

            if val_loader is not None:
                # evaluate model on validation data
                for (Xv, yv, _) in val_loader:
                    # squeeze necessary (altern...change trainloader (to do))
                    if len(yv.shape) > 2:
                        yv = yv.squeeze(1)
                    #Xv = torch.Tensor(Xv).to(self.device)
                    Xv = Xv.to(self.device)
                    yv_B, xv_B, _, _ = self.model(Xv)
                    lossv, loss_componentsv = self.loss_fct(yv_B, xv_B, yv, Xv, losswgt_in, self.sub_losses)
                    # add up mini batch losses
                    epch_loss_val += lossv.item() / len(val_loader)
                    # loss components
                    for k in range(0, len(self.sub_losses)):
                        component_losses_val[k] += loss_componentsv[k].item() / len(val_loader)
                all_epoch_losses_val.append(component_losses_val + [epch_loss_val] + [0.0] )

            print('SAE train: Epoch: ', epoch, ' - loss weights: ', losswgt_in)  # params[-len(lossweights):]

        self.all_losses = all_epoch_losses
        self.all_losses_val = all_epoch_losses_val

    # test data predictions
    def predict_H(self,
                  X_in: np.array,
                  pd_index: pd.Index,
                  colnames_list: List[str] = ['H'] ) -> pd.DataFrame:
        '''
        prediction of the low dimension "H" layer
        :param X_in: input (sentiment data X (t x K topics)
        :param pd_index: provide index to attach dates to pandas df
        :param colnames_list: provide list of column names for H (order must match the losses)
        :param trainloader: torch data loader
        :param testloader: torch test data loader
        :return:
        '''
        self.model.eval()

        X = torch.tensor(X_in, dtype = torch.float) # customize this
        with torch.no_grad():
            # dont train, use trained model
            self.model.to(self.device)
            y,x,H,_ = self.model(X)

        pd_out = pd.DataFrame(H.cpu().detach().numpy().squeeze(), index=pd_index)
        pd_out.columns = colnames_list

        if self.classification:
            # change sign of H (have 0=negative, 1=positive) if linear mapping configured as 1=negative, 0=positive
            for i in range(0,len(self.model.enc.cl1.transforms)):
                # check directionality in classification mapping:
                # need first - then + to match 0,1 labels
                if self.model.enc.cl1.transforms[i].weight.data[0] > 0 and self.model.enc.cl1.transforms[i].weight.data[1] < 0:
                    pd_out[pd_out.columns[i]] = pd_out[pd_out.columns[i]]*-1

        return pd_out


