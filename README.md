
# Code repo "news representation"
Accompanies paper "Learning Deep news representations for macro finance". 

Implements pytorch versions of supervised autoencoder (SAE) architectures (see
Le, L., Patterson, A., & White, M. (2018)) in the financial context of the paper.

sae_start.py implements an example run for an example data set in the context of the paper.
This includes all four competing models.
repr_nb.ipynb contains a notebook with an SAE example. 

The utils folder contains the bulk of the code for the neural (+linear) representation retrieval:
 
- data loaders
- loss functions
- models 
- sae class with train and test functionality 
- PLS / rolling analysis: linear tools

The applications folder contains the code used in the applications sections of the paper. 

- applications_panel.py contains the panel analysis
- applications_tsm.py contains the trend strategy
- midas_repr.R contains the nowcasting analysis



Early version of SAE implementation based off of https://github.com/mortezamg63/Supervised-autoencoder











Le, L., Patterson, A., & White, M. (2018). Supervised autoencoders: Improving generalization performance with unsupervised regularizers. Advances in neural information processing systems, 31, 107-117.

## A Note on the Hidden Representation Extraction

Note that the specific architectures can induce sign flips from the 'hidden representation of interest' to the supervision variable. This is due to the last linear map.  When extracting the hidden representations,
the 
'predict_H' - function takes care of reconciling the signs of the hidden representations with the supervisions.