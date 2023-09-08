import torch
from typing import List, Tuple

def loss_UW(
        y_hat: torch.Tensor,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        log_lossweights: List[torch.Tensor],
        losses: List[torch.nn.Module]
    ) -> Tuple[ torch.Tensor, List[torch.Tensor]] :
    '''

    not used ... DEPRECATED
    idea: learn loss weights along the way

    uncertainty weight loss
    -> use case: learn weights of losses while training

    :param y_hat: hidden layer "predictions" for additional losses
    :param x_hat: final layer "predictions" for the standard reconstruction loss
    :param y: supervision of y_hat
    :param x: supervision of x_hat ( the initial data fed in )
    :param log_lossweights:  list of log weights for losses
    :param losses: list of losses: [ nn.NLLLoss() , nn.L1Loss() ]
    :return: tuple of scalar loss and list of sublosses
    '''


    factor_1 = torch.div(1.0, torch.mul(2.0, log_lossweights[0].exp().pow(2)))
    l1 = losses[0](x_hat, x)  # .squeeze()
    l = torch.add(torch.mul(factor_1, l1), torch.log(log_lossweights[0].exp()))
    sublosses  = [l]

    for i in range(1,len(log_lossweights)):
        factor_i = torch.div(1.0, torch.mul(2.0, log_lossweights[i].exp().pow(2)))
        # y broadcasted if not matching y_hat! :
        l_i = losses[i](y_hat[:,i-1], y[:,i-1])  # dims y_hat: B x encodingdim   dims y: B x nr of losses
        l_i = torch.add(torch.mul(factor_i, l_i), torch.log(log_lossweights[i].exp()))
        sublosses.append( l_i )
        l = l.add( l_i )

    return l, sublosses


# same as fixedW loss, only for RNN
def loss_fixedW_RNNencdec(
        y_hat: torch.Tensor,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        log_lossweights: List[ torch.Tensor ],
        losses: List[ torch.Tensor ]
    ) -> Tuple[ torch.Tensor, List[ torch.Tensor ]]:
    '''
    fixed weight loss ... use case: dont want to learn loss weights, these are fixed. final loss scalar
    is simple weighted (by lossweights) average of sublosses

    :param y_hat: hidden layer "predictions" for additional losses
    :param x_hat: final layer "predictions" for the standard reconstruction loss
    :param y: supervision of y_hat
    :param x: supervision of x_hat ( the initial data fed in )
    :param log_lossweights:  list of log weights for losses
    :param losses: list of losses: [ nn.NLLLoss() , nn.L1Loss() ]
    :return: tuple of scalar loss and list of sublosses
    '''
    # reconstruction loss only against last element
    l = torch.mul(log_lossweights[0], losses[0](x_hat, x[:,-1,:]) )
    sublosses  = [l]

    # note: first loss is reconstruction
    for i in range(1,len(log_lossweights)):

        # y broadcasted if not matching y_hat! :
        l_i = losses[i](y_hat[:,i-1], y[:,i-1])  # dims y_hat: B x encodingdim   dims y: B x nr of losses

        l_i = torch.mul( log_lossweights[i], l_i)
        sublosses.append( l_i )
        l = l.add( l_i )

    return l, sublosses



def loss_fixedW(
        y_hat: torch.Tensor,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        x:torch.Tensor,
        log_lossweights: List[torch.Tensor],
        losses: List[torch.Tensor]
    ) -> Tuple[ torch.Tensor, List[ torch.Tensor ]] :
    '''
    fixed weight loss ... use case: dont want to learn loss weights, these are fixed. final loss scalar
    is simple weighted (by lossweights) average of sublosses

    :param y_hat: hidden layer "predictions" for additional losses
    :param x_hat: final layer "predictions" for the standard reconstruction loss
    :param y: supervision of y_hat
    :param x: supervision of x_hat ( the initial data fed in )
    :param log_lossweights:  list of log weights for losses
    :param losses: list of losses: [ nn.NLLLoss() , nn.L1Loss() ]
    :return: tuple of scalar loss and list of sublosses
    '''

    # the reconstruction loss
    l = torch.mul(log_lossweights[0], losses[0](x_hat, x) )
    sublosses  = [l]

    # note: first loss is reconstruction
    for i in range(1,len(log_lossweights)):
        # y broadcasted if not matching y_hat
        l_i = losses[i](y_hat[:,i-1], y[:,i-1])  # dims y_hat: B x encodingdim   dims y: B x nr of losses

        l_i = torch.mul( log_lossweights[i], l_i)
        sublosses.append( l_i )
        l = l.add( l_i )

    return l, sublosses



# loss for shallow autoencoder / PCA
def loss_shallow(
        y_hat: torch.Tensor,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        log_lossweights: List[torch.Tensor],
        losses: List[torch.Tensor] ) -> Tuple[ torch.Tensor, List[torch.Tensor]]:
    # same as above
    return losses[0](x_hat, x), [torch.tensor(0.0) for x in range(0,len(losses))]



def loss_fixedW_noreconstruction(
        y_hat: torch.Tensor,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        log_lossweights: List[ torch.Tensor ],
        losses: List[ torch.Tensor ]
    ) -> Tuple[ torch.Tensor, List[torch.Tensor]]:
    '''
    fixed weight loss ... use case: dont want to learn loss weights, these are fixed. final loss scalar
    is simple weighted (by lossweights) average of sublosses

    :param y_hat: hidden layer "predictions" for additional losses
    :param x_hat: final layer "predictions" for the standard reconstruction loss
    :param y: supervision of y_hat
    :param x: supervision of x_hat ( the initial data fed in )
    :param log_lossweights:  list of log weights for losses
    :param losses: list of losses: [ nn.NLLLoss() , nn.L1Loss() ]
    :return: tuple of scalar loss and list of sublosses
    '''


    sublosses  = [ torch.as_tensor( 0.0 ) ]

    for i in range(1,len(log_lossweights)):

        # y broadcasted if not matching y_hat! :
        l_i = losses[i](y_hat[:,i-1], y[:,i-1])  # dims y_hat: B x encodingdim   dims y: B x nr of losses

        l_i = torch.mul( log_lossweights[i], l_i)
        sublosses.append( l_i )

        if i==1:
            l = l_i
        else:
            l = l.add( l_i )

    return l, sublosses


# collect absolute values of activations
def sparse_loss(
        activation_list: List[torch.Tensor]
    )-> torch.Tensor:
    '''
    :param activation_list: list of activations (encoder, bottleneck, decoder)
    :return: summed up abs values
    '''
    loss = 0
    for i in range(len(activation_list)):
        loss += torch.mean(torch.abs(activation_list[i]))
    return loss