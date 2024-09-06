import numpy as np
import torch.nn as nn  # construct NN
import torch.nn.functional as F
import torch
from typing import Dict, Literal,  Tuple, List, Union



# custom module handling elementwise losses for hidden layer of interest
class HiddenElementsFinalLayer(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 map_element_size: int
                 ):
        super(HiddenElementsFinalLayer, self).__init__()
        # map from 1 element to 2
        self.transforms = nn.ModuleList([nn.Linear(1,map_element_size) for _ in range(hidden_size)])

    def forward(self,
                x:  torch.Tensor
                ) -> torch.Tensor:
        x_split = torch.split(x,1,1) # hidden_size elements
        res = []
        for i,transform in enumerate(self.transforms):
            res.append( transform(x_split[i]) )
        return torch.cat(res,dim = 1)



# custom module handling elementwise losses for hidden layer of interest
# fully connected version
class HiddenElementsFinalLayer_fullyconnected(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 map_element_size: int
                 ):
        super(HiddenElementsFinalLayer_fullyconnected, self).__init__()
        # map from 1 element to 2
        self.transforms = nn.ModuleList([nn.Linear(hidden_size,map_element_size) for _ in range(hidden_size)])

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        res = []
        for i,transform in enumerate(self.transforms):
            res.append( transform(x) )
        return torch.cat(res,dim = 1)




# shallow to deep encoding
class encoder_netw(nn.Module):
    def __init__(self,
                 X_dim_in: int,
                 map_layer_dims: List[int] = [20],
                 encoding_layer_dim: int=10,
                 dropout_p: float = 0.2,
                 classification: bool = True,
                 activations: Dict[str, nn.Module ] = {'initial' : nn.Tanh(), 'H' : nn.Tanh(), 'loss' : nn.Tanh() },
                 fc_last_lyr: bool = False ):

        # inherit attributes and methods of nn.Module
        super(encoder_netw, self).__init__()
        layer_dims = [X_dim_in] + map_layer_dims + [encoding_layer_dim]
        self.linear_transforms = nn.ModuleList([nn.Linear(layer_dims[i],layer_dims[i+1]) for i in range((len(layer_dims)-1))] )
        # batch norm
        #self.BN = nn.ModuleList([ nn.BatchNorm1d(num_features=layer_dims[i+1]) for i in range((len(layer_dims)-1)) ] )
        self.activ_initial = activations['initial']  # activation
        self.activ_H = activations['H']  # activation
        self.activ_loss = activations['loss']  # activation
        if fc_last_lyr:
            finallayer = HiddenElementsFinalLayer_fullyconnected
        else:
            finallayer = HiddenElementsFinalLayer
        #if classification:
        #    self.cl1 = finallayer(hidden_size=encoding_layer_dim, map_element_size=1)
        #else:
        #    self.cl1 = finallayer(hidden_size=encoding_layer_dim, map_element_size=1)
        self.cl1 = finallayer(hidden_size=encoding_layer_dim, map_element_size=1)

        self.dropout_p = dropout_p
        self.encoding_layer_dim = encoding_layer_dim

    def forward(self,
                x: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activations_output = [] # collect activations for sparse autoencoder loss functions
        for k, lin_transform in enumerate(self.linear_transforms[0:-1]):
            x = lin_transform(x) # transform
            #x = self.BN[k](x)  # batch normalize
            x = F.dropout(self.activ_initial(x), p = self.dropout_p, training=self.training)  # dropout + activation fct
            activations_output.append(x)  # collect activations for sparse ae loss
        # no dropout for final encoding
        x = self.linear_transforms[-1](x)
        #x = self.BN[-1](x)
        x = self.activ_H( x )
        activations_output.append(x)
        # final transform gets fed to loss
        y_hat = self.activ_loss(self.cl1(x)) # activ_loss is activation directly before feeding to loss
        return y_hat, x, activations_output  # y_hat, H



# shallow to deep decoding
class decoder_netw(nn.Module):
    def __init__(self,
                 X_dim_in: int,
                 map_layer_dims: List[int] = [20],
                 encoding_layer_dim: int = 10,
                 dropout_p: float = 0.2,
                 activations: Dict[str, nn.Module ]  = {'initial': nn.Tanh(), 'final': nn.Identity() } ):
        # inherit attributes and methods of nn.Module
        super(decoder_netw, self).__init__()
        layer_dims = [X_dim_in] + map_layer_dims + [encoding_layer_dim]
        self.linear_transforms = nn.ModuleList([nn.Linear(layer_dims[i],layer_dims[i-1]) for i in range((len(layer_dims)-1),0,-1)] )
        #self.BN = nn.ModuleList([nn.BatchNorm1d(num_features=layer_dims[i-1]) for i in range((len(layer_dims)-1),1,-1)])  # no BN for final output
        self.activ_initial = activations['initial']
        self.activ_final = activations['final']
        self.dropout_p = dropout_p
        self.encoding_layer_dim = encoding_layer_dim

    def forward(self,
                x: torch.Tensor) -> Tuple[ torch.Tensor, torch.Tensor ] :
        activations_output = []  # collect activations for sparse autoencoder loss functions
        for k, lin_transform in enumerate(self.linear_transforms[0:-1]):
            x = lin_transform(x)
            #x = self.BN[k](x)  # batch normalize
            x = F.dropout(self.activ_initial(x), p=self.dropout_p, training=self.training)
            activations_output.append(x)
        # no dropout, no activation for final output
        return self.activ_final( self.linear_transforms[-1](x) ), activations_output  # final decoded rep, activations for sparse ae loss



# LSTM encoder
# to be done
class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 classification: bool = True,
                 drop_prob: float = 0.,
                 rnn: Union[nn.GRU, nn.LSTM]  = nn.GRU,
                 activations: Dict[str, nn.Module ]  =
                    {'initial' : nn.Tanh(), 'H' : nn.Tanh(), 'loss' : nn.Tanh() },
                 fc_last_lyr: bool = False ):
        super(RNNEncoder, self).__init__()
        self.dropout_p = drop_prob

        self.rnn = rnn(input_size, hidden_size, num_layers,
                          batch_first=True,bidirectional=False,
                          dropout=drop_prob if num_layers > 1 else 0.)
        if fc_last_lyr:
            finallayer = HiddenElementsFinalLayer_fullyconnected
        else:
            finallayer = HiddenElementsFinalLayer
        #if classification:
        #    self.cl1 = finallayer(hidden_size=hidden_size, map_element_size=2)
        #else:
        #    self.cl1 = finallayer(hidden_size=hidden_size, map_element_size=1)
        self.cl1 = finallayer(hidden_size=hidden_size, map_element_size=1)
        self.activ_loss = activations['loss']  # activation of pre-final layer

    def forward(self,
                x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # LSTM output:  output, (h_n, c_n)
        # _,test = self.model._modules['enc'].rnn(X)
        _, H = self.rnn(x)
        # H shape is (directions *num_layers, B, hiddensize )
        H = torch.mean(H, 0, keepdim=True) # get rid of layers -> could use linear layer instead
        H = H.squeeze(0)  # get rid of non-batch singleton dims
        # Apply dropout (RNN applies dropout after all but the last layer)
        H = F.dropout(H, self.dropout_p, self.training)
        #
        # final transform gets fed to loss
        y_hat = self.activ_loss(self.cl1(H))  # activ_loss is activation directly before feeding to loss
        return y_hat, H, [torch.zeros(2,3)] # no activations returned

class TransformerEncoder(nn.Module):

    def __init__(self,
                 input_size : int,
                 hidden_size : int,
                 dim_model : int = 64,
                 dim_ff : int = 128,
                 n_heads : int = 1,
                 num_layers : int = 8,
                 dropout : float = 0.1,
                 activations : Dict[str, nn.Module] = dict(H = nn.LeakyReLU(0.2), loss = nn.Sigmoid()),
                 positional_encoding : Literal['none', 'sinusoidal', 'trainable'] = 'sinusoidal',
                 max_input_len : int = 1024
                 ):
        
        super().__init__()

        self.embed = nn.Linear(input_size, dim_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = dim_model,
                                                                            dim_feedforward = dim_ff,
                                                                            nhead = n_heads,
                                                                            dropout = dropout,
                                                                            batch_first = True,
                                                                            activation = 'gelu'),
                                                 num_layers = num_layers)
        self.unembed = nn.Linear(dim_model, hidden_size)
        self.activation = activations['H']
        self.head = HiddenElementsFinalLayer(hidden_size, map_element_size = 1)
        self.final_act = activations['loss']

        if positional_encoding == 'none':
            self.pos_enc = nn.Parameter(torch.zeros(max_input_len, dim_model), requires_grad = False)
        elif positional_encoding == 'sinusoidal':
            def calc_pos_enc(i, j, N = 10000):
                n = N ** (1 / dim_model)
                if j % 2 == 0:
                    return np.sin(i / (n**j))
                else:
                    return np.cos(i / (n**(j-1)))
            pos_enc = torch.empty(max_input_len, dim_model)
            for i in range(max_input_len):
                for j in range(dim_model):
                    pos_enc[i, j] = calc_pos_enc(i, j)
            self.pos_enc = nn.Parameter(pos_enc, requires_grad = False)
        elif positional_encoding == 'trainable':
            self.pos_enc = nn.Parameter(torch.empty(max_input_len, dim_model), requires_grad = True)
            self.init_weights(self.pos_enc)
        else:
            raise ValueError(f'Positional encoding keyword \'{positional_encoding}\' unrecognised.')

        self.apply(self.init_weights)
    
    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        pos_enc = torch.tile(self.pos_enc[:seq_len], (batch_size, 1, 1))

        H = self.transformer(self.embed(x) + pos_enc)
        H = self.activation(self.unembed(H))

        y_hat = self.final_act(self.head(H))
        return y_hat, H, [torch.zeros(2, 3)]

    @staticmethod
    def init_weights(m : nn.Module):
        try:
            nn.init.trunc_normal_(getattr(m, 'weight', m), mean = 0, std = 0.02)
        except AttributeError:
            if hasattr(m, 'bias'):
                m.bias.fill_(0.)
        
# wrappers:
# encoder decoder setting
class encoder_decoder(nn.Module):
    # encoder decoder init
    def __init__(self,
                 X_dim_in: int,
                 map_layer_dims: List[int] = [20],
                 encoding_layer_dim: int = 10,
                 dropout_p: float = 0.2,
                 classification: bool = True,
                 activations: Dict[str, nn.Module ]  =
                    {'initial': nn.Tanh(), 'H': nn.Tanh(), 'loss': nn.Tanh(), 'final': nn.Identity()},
                 external_enc = Union[nn.Module,None] ):
        super(encoder_decoder, self).__init__()
        if external_enc is None:
            self.enc = encoder_netw(X_dim_in, map_layer_dims, encoding_layer_dim, dropout_p, classification,activations )
        else:
            self.enc = external_enc
        self.dec = decoder_netw( X_dim_in, map_layer_dims, encoding_layer_dim, dropout_p, activations )

    def forward(self,
                x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor] ]:
        y_hat, H, activations_output_enc = self.enc(x)
        x_hat, activations_output_dec = self.dec(H)
        activations_output_enc += activations_output_dec
        return y_hat, x_hat, H, activations_output_enc



# pure encoder, x mapped back unchanged, allow LSTM encoding to enter this
class encoder(nn.Module):
    # encoder decoder init
    def __init__(self,
                 X_dim_in: int,
                 map_layer_dims: List[int] = [20],
                 encoding_layer_dim: int = 10,
                 dropout_p: float = 0.2,
                 classification: bool = True,
                 activations: Dict[str, nn.Module ]  =
                    {'initial': nn.Tanh(), 'H': nn.Tanh(), 'loss': nn.Tanh(), 'final': nn.Identity()},
                 external_enc: Union[nn.Module, None]= None
                 ):
        super(encoder, self).__init__()
        if external_enc is not None:
            self.enc = external_enc
        else:
            self.enc = encoder_netw(X_dim_in, map_layer_dims, encoding_layer_dim, dropout_p, classification, activations)


    def forward(self,
                x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[ torch.Tensor ]]:
        y_hat, H, activations_output_enc = self.enc(x)
        return y_hat, x, H, activations_output_enc

