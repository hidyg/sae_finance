import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.losses import sparse_loss
# tensorboard
from tensorboardX import SummaryWriter
# import tensorflow as tf
from typing import Callable, List, Tuple, Union




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
                 sparse_regularizer: float = 0.0,
                 tx_writer: SummaryWriter =  SummaryWriter('runs'),
                 save_model = False
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
        :param l1strength: L1 reg on weights (not used)
        :param train_loss_wgt:  train loss weights along the way
        :param classification: switch on the type of additional losses ( MSE possible. currently: should be True, want to do classification)
        :param sparse_regularizer: possibility to use sparse autoencoders. this is the penalty term multiplier
        :param tx_writer: tensorboardX summarywriter

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
        self.tx_writer = tx_writer # Saves the summaries to the directory 'runs' in the current parent directory
        self.save_model = save_model

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

            # in the firsttrain_reconstruction epochs, only train the reconstruction loss
            # only classification case so far
            if not self.train_loss_wgt and epoch<self.firsttrain_reconstruction and self.classification:
                losswgt_in = lossweights_first_epochs
            else:
                losswgt_in = lossweights

            for (X, y, idxs) in train_loader:
                batchidx += 1

                X = X.to(self.device)
                y = y.to(self.device)

                # squeeze necessary (altern...change trainloader (to do))
                if len(y.shape) > 2:
                    y = y.squeeze(1)

                optimizer.zero_grad()
                y_hat, x_hat, _, activations_out = self.model(X)
                loss, loss_components = self.loss_fct(y_hat, x_hat, y.float(), X, losswgt_in, self.sub_losses)

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
                        #_, pred_class = torch.max(y_hat[:,(i*2):(i*2+2)].data, 1)  # returns tuple
                        pred_class = (y_hat[:, i].data > torch.tensor(0.5)).int()  #
                        # add up mini batch acc
                        epch_acc[i] += (pred_class.unsqueeze(1) == y[:,i].unsqueeze(1) ).sum().item() / (len(train_loader) * X.size(0))

                # loss components
                for k in range(0, len(self.sub_losses)):
                    component_losses[k] += loss_components[k].item() / len(train_loader)

            if self.save_model:
                # to do, save after each epch
                pass

            #  collect losses
            all_epoch_losses.append( component_losses + [epch_loss] + epch_acc )
            print('SAE train: Batch Nr', batchidx, ' iteration')
            print('SAE train: Epoch: ', epoch, ' - train_loss: ', epch_loss)
            print('SAE train: Epoch: ', epoch, ' - train_accuracies  (if classif.): ', epch_acc)
            printout = 'SAE train: Epoch: '+ str( epoch )
            printout = printout + '. '.join([' - loss_' +str(lnr)+':' + str(component_losses[lnr])   for lnr in range(0,len(self.sub_losses))])
            print(printout) #'SAE train: Epoch: ', epoch, ' - loss_1: ', component_losses[0], ' - loss_2: ', component_losses[1]

            # write tensorboardX summary
            self.write_tx_summaries( run_type='(training)', component_losses=component_losses,
                                    epch_loss=epch_loss, epch_acc=epch_acc, e=epoch)

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

                # tensorboardX summary
                self.write_tx_summaries(run_type='(validation)',component_losses=component_losses_val,
                                        epch_loss = epch_loss_val,epch_acc=[], e = epoch)

            print('SAE train: Epoch: ', epoch, ' - loss weights: ', losswgt_in)  # params[-len(lossweights):]

        self.all_losses = all_epoch_losses
        self.all_losses_val = all_epoch_losses_val


    def write_tx_summaries(self,
                           run_type: str,
                           component_losses: List[float],
                           epch_loss: List[float],
                           epch_acc: List[float],
                           e
                           ) -> None:
        '''
        :param tx_writer: initialized tx writer
        :param run_type: val / test
        :param component_losses:  list of individual losses
        :param epch_loss:  aggregate loss
        :param epch_acc:   accuracies for supervision losses
        :param e:  epoch nr
        :return: none
        '''
        # write tensorboardX summary validation
        for k in range(0, len(component_losses)):
            self.tx_writer.add_scalar(run_type+' component loss ' + str(k), component_losses[k], e)
        self.tx_writer.add_scalar(run_type+' final loss sum', epch_loss, e)
        for k in range(0, len(epch_acc)):
            self.tx_writer.add_scalar(run_type+' component accuracy ' + str(k), epch_acc[k], e)


    # test data predictions
    def predict_H(self,
                  X_in: np.array,
                  pd_index: pd.Index,
                  colnames_list: List[str] = ['H'],
                  trainloader:  Union[torch.utils.data.DataLoader,None] = None,
                  testloader:  Union[torch.utils.data.DataLoader,None] = None) -> pd.DataFrame:
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

        X = torch.tensor(X_in, dtype = torch.float, device=self.device) # customize this
        if trainloader is None and testloader is None:
            with torch.no_grad():
                # dont train, use trained model
                self.model.to(self.device)
                y,x,H,_ = self.model(X)
        else:
            with torch.no_grad():
                # collect data by running through train and test loaders
                y_out, x_out, H_out = [],[],[]
                if trainloader is not None:
                    for (X, _, _) in trainloader:
                        X = X.to(self.device)
                        y_B, x_B, H_B, _ = self.model(X)
                        y_out.append( y_B )
                        x_out.append( x_B )
                        H_out.append( H_B )
                if testloader is not None:
                    for (X, _, _) in testloader:
                        X = X.to(self.device)
                        y_B, x_B, H_B, _ = self.model(X)
                        y_out.append(y_B)
                        x_out.append(x_B)
                        H_out.append(H_B)

            # the predictions
            y, x, H = torch.cat(y_out), torch.cat(x_out), torch.cat(H_out)

        pd_out = pd.DataFrame(H.cpu().detach().numpy().squeeze(), index=pd_index)
        pd_out.columns = colnames_list

        # align signs in the H time series to match supervisions
        if self.classification:
            # change sign of H (have 0=negative, 1=positive) if linear mapping configured as 1=negative, 0=positive
            for i in range(0,len(self.model.enc.cl1.transforms)):
                # check directionality in classification mapping:
                # need first - then + to match 0,1 labels
                #if self.model.enc.cl1.transforms[i].weight.data[0] > 0 and self.model.enc.cl1.transforms[i].weight.data[1] < 0:
                if self.model.enc.cl1.transforms[i].weight.data[0] < 0:
                    pd_out[pd_out.columns[i]] = pd_out[pd_out.columns[i]]*-1
                    print('signs flipped')

        return pd_out


