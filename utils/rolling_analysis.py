import pandas as pd
import numpy as np
from sklearn.utils.extmath import svd_flip



# class PCA
class simplePCA:
    '''
    simple PCA with optionality for "U-based", "V-based" SVD, "negative" eigenvector handling
    '''
    def __init__(self, PCAtype = 'cov', nr_components = 2, svd_flip_this = False):
        '''
        :param df: pandas dataframe based on numeric data to perform PCA on (timestamp index included)
        :param PCAtype: 'cor' or 'cov' - correlation or covariance type PCA
        :param nr_components: nr of components to extract
        '''
        self.PCAtype = PCAtype
        self.components = nr_components
        self.svd_flip_this = svd_flip_this
        self.eigen_values = None
        self.eigen_vectors = None
        self.variance_explained = None

    def fit_pca( self, df_in ):
        '''
        :param df_in: pandas in
        :return: None, fill slots of instance
        '''
        X = df_in.to_numpy()
        # centering ? -> cor, cov or data already centered and properly scaled
        if self.PCAtype == 'cov':
            # each row is variable
            print('cov used (variable scales are similar)')
            covariance_matrix = np.cov( X.T )
        elif self.PCAtype == 'cor':
            print('cor used (variables are on different scales)')
            # each row is variable
            covariance_matrix = np.corrcoef( X.T )
        else: # else .... no cov!
            print(" X'X used (data already centered & scaled)")
            covariance_matrix = np.matmul(X.T,X)
        # get svd
        # u*s*vh     V= vh convention in the following
        U, S, V = np.linalg.svd(covariance_matrix, hermitian = True)  #svd(covariance_matrix) #
        idx = np.argsort(-S) # sort eigenvalues descending
        # re-order U and V
        if self.svd_flip_this:
            # order
            print('flip U and V')
            U, V = svd_flip(U, V, u_based_decision= True )

        # eigenvectors form rows of vh (V)
        V =  V[ idx , :  ]
        S = S[idx]
         # yields idx of eigen values sorted dec
        self.eigen_values, self.eigen_vectors = S, V

        # explained variance
        self.variance_explained = []
        for i in self.eigen_values:
            self.variance_explained.append((i / sum(self.eigen_values)) * 100)

    def pca_transform(self, df_in, out_type = 'pandas', targetvol_ann = None, automatic_sign_change = False):
        '''
        :param df_in: pandas df
        :param out_type:  'pandas' or numpy array
        :param targetvol_ann:  annualized target vol of PCs if needed for plotting
        :return: pd df or np array
        '''
        X = df_in.to_numpy()
        #projection_matrix = (self.eigen_vectors.T[:][:self.components]).T
        #print('new2')
        M_eigen = self.eigen_vectors[:self.components, ]
        if automatic_sign_change:
            avs = M_eigen.mean(axis = 1)
            #neg_fraction = (M_eigen<0).sum(axis = 1) / M_eigen.shape[1]
            M_eigen[ avs<0.0,:] = M_eigen[avs<0.0,:] * -1
        projection_matrix = M_eigen.T  # now cols transformed
        #
        # Getting the product of original standardized X and the eigenvectors
        X_pca = X.dot(projection_matrix)
        if targetvol_ann is not None:
            for i in range(0,X_pca.shape[1]): X_pca[:,i] = X_pca[:,i] / (X_pca[:,i].std() * np.sqrt(252)) * targetvol_ann
        if out_type == 'pandas':
            pd_pca = pd.DataFrame(X_pca, columns=['PC' + str(x) for x in range(0,self.components)], index = df_in.index)
            return pd_pca
        else:
            return X_pca


    # expanding window rolling pca
    def rolling_fit_transform_xpdng(self, df_in, out_type = 'pandas', targetvol_ann = None, automatic_sign_change = False,
                              OOS_periods = 12, start_period = 36):
        '''
        :param df_in: pandas df
        :param out_type: pandas or np array
        :param targetvol_ann: scale cols after fit -> None or float
        :param automatic_sign_change: apply change according to eigenvalues
        :param OOS_periods: nr of out-of-sample periods, e.g. 12 if on monthly freq
        :param start_period: nr of last observation for the first in-sample period (int)
        :return: df or numpy
        '''
        #
        # split df_in
        splits =  [i for i in range(start_period,df_in.shape[0],OOS_periods)]
        if splits[-1]!=df_in.shape[0]:
            splits += [df_in.shape[0]]

        # results
        pca_res = []
        for k in range(0,(len(splits)-1)):
            # get sub df
            if k>0:
                df_oos = df_in.iloc[splits[k]:splits[k+1]]
            else:
                df_oos = df_in.iloc[0:splits[k + 1]]
            df_sub = df_in.iloc[0:splits[k]]
            # do mini pca
            pca_sub = simplePCA(PCAtype=self.PCAtype, nr_components=self.components)
            pca_sub.fit_pca(df_sub)
            # no vol scaling here
            df_transformed_sub = pca_sub.pca_transform(df_oos, out_type = out_type,
                                                       targetvol_ann= None, automatic_sign_change=automatic_sign_change)
            # collect results
            pca_res.append(df_transformed_sub)


        # get the final and scale
        if out_type == 'pandas':
            df_oos_rolling = pd.concat(pca_res, axis = 0)
            if targetvol_ann is not None:
                for c in df_oos_rolling.columns:
                    df_oos_rolling[c] = df_oos_rolling[c] / (df_oos_rolling[c].std() * np.sqrt(252)) * targetvol_ann
        else:
            df_oos_rolling = np.concatenate(pca_res, axis = 0)
            if targetvol_ann is not None:
                for i in range(0, df_oos_rolling.shape[1]):
                    df_oos_rolling[:, i] = df_oos_rolling[:, i] / (df_oos_rolling[:, i].std() * np.sqrt(252)) * targetvol_ann

        return df_oos_rolling
