import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LassoCV



# PLS, log regression, Lasso
class PLS:
    def __init__(self, nr_components = 5, model_type = 'PLS' ):
        '''
        :param nr_components: PLS components
        :param classification: determines if accuracies important
        :param model_type: 'PLS' or 'logreg' (logistic regression) or Lasso (5 fold CV for alpha selection)
        '''
        if model_type == 'PLS':
            self.pls = [PLSRegression(n_components = 1) for i in range(0,nr_components)]
        elif model_type == 'logreg':
            self.pls = [LogisticRegression(random_state = 0 ) for i in range(0, nr_components)]
        elif model_type == 'logregL1':
            self.pls = [LogisticRegressionCV(random_state = 0, cv=4, penalty='l1', solver='saga', max_iter = 25 )
                        for i in range(0, nr_components)]
        elif model_type == 'Lasso':
            self.pls = [LassoCV(cv=5) for i in range(0, nr_components)]
        self.model_type = model_type
        self.nr_comp = nr_components

    def train(self, traindata):
        '''
        :param traindata: from torch dataloader
        :return: modifies pls slot
        '''
        if self.model_type in ['logregL1']:
            # need binary Y
            for i in range(0,len(self.pls)):
                if np.sum(traindata.Ys[:, i])==0:
                    traindata.Ys[ 10 , i] = 1 # add arbitrary 1 to make things work
        self.pls = [x.fit( traindata.Xs, traindata.Ys[:,i] ) for i,x in enumerate(self.pls)]

    def predict_H(self, X, pd_index, colnames_list):
        '''
        :param X: train / test data
        :param pd_index: index to be used for pandas construction
        :param colnames_list:  names of H cols
        :return:  pandas output containing H
        '''
        # collect all representations
        if not self.model_type in ['Lasso','logregL1']:
            for i,p in enumerate(self.pls):
                H = p.predict( X )
                if i==0:
                    H_out = [H]
                else:
                    H_out.append(H)
        else:
            for i,p in enumerate(self.pls):
                H = p.predict( X )
                if i==0:
                    H_out = [np.expand_dims(H,axis=1)]
                else:
                    H_out.append(np.expand_dims(H,axis=1))

        H = np.concatenate(H_out, axis = 1)

        pd_out = pd.DataFrame(H, index=pd_index)
        pd_out.columns = colnames_list
        return pd_out


