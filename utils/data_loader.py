# own data loader for sent data frames
import torch
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any, Union, Tuple, Type


class sent_Dataset_general(torch.utils.data.Dataset):
    """ general sent data set class """
    def __init__(self,
                 df : pd.DataFrame,
                 df_cutoff : datetime.datetime,
                 train : bool =True,
                 Y_names : List[str]  = ['EQidx_S&P500Mini'],
                 Y_name_types : Dict[str,Any ] = {'EQidx_S&P500Mini': int},
                 X_names : List[str] = ['revenues_ESS', 'security_ESS', 'social-relations_ESS',
                            'pollution_ESS', 'production_ESS', 'products-services_ESS'],
                 #Date_name = 'DATE',
                 train_dates: Union[pd.DatetimeIndex, None] = pd.date_range(start="2001-01-01",end="2018-12-31"),
                 test_dates: Union[pd.DatetimeIndex, None] = None,
                 convert_Y_to_labels : bool = True,
                 seq_len : int = 21,
                 lead_Y: int = 0):
        '''
        :param df: input data frame containing X and Y
        :param df_cutoff: discard data below this index value (date)
        :param train: train or test version (T/F)
        :param Y_names: names of ys
        :param Y_name_types: name types (int or np.float32)
        :param X_names: names of xs
        :param train_dates: specific dates for the training set
        :param test_dates: the specific dates for the test set
        :param convert_Y_to_labels: convert Y to 0/1 for classification or leave as is
        :param seq_len: sequence length of data items (for subsequent RNN use)
        :param lead_Y: lead or lag the Y (int )
        '''

        # nr of ys:
        self.nrys = len(Y_names)
        self.sequence_length = seq_len

        # read df
        #df.set_index(pd.Index(pd.to_datetime(df[Date_name], format='%Y-%m-%d'), name="DATE"), inplace=True, drop =True)
        df = df.loc[df.index > df_cutoff]
        df.fillna(0, inplace=True)

        df[X_names] = df[X_names].astype(np.float32)

        for k in Y_name_types.keys():
            # implement lags / leads
            if lead_Y != 0:
                df[k] = df[k].shift(lead_Y)
                df[k] = df[k].fillna(0.0)
            # to binary for classification
            if convert_Y_to_labels:
                df[k] = (df[k]>0)*1
            # change data type to required type
            df.loc[:,k] = df.loc[:,k].astype(Y_name_types[k])


        # set training and test data size
        self.train = train

        # specific test dates provided?
        if test_dates is None:
            test_dates_ind = np.logical_not(np.in1d(df.index, train_dates)) # index of test dates remainder of dates
        else:
            test_dates_ind = np.in1d(df.index, test_dates)  # index of test dates

        # train / test specs different
        if self.train:
            self.Xs = df.loc[np.in1d(df.index,train_dates), X_names].values   #.astype(np.float32)
            self.pdidx = df.index[np.in1d(df.index,train_dates)]
            self.Ys = df.loc[np.in1d(df.index, train_dates), Y_names].values  #.astype(Y_name_types)
            print("Training on {} examples".format(np.sum(np.in1d(df.index,train_dates))) )
        else:
            self.Xs = df.loc[test_dates_ind, X_names].values  #.astype(np.float32)
            self.pdidx = df.index[test_dates_ind]
            self.Ys = df.loc[test_dates_ind, Y_names].values  #.astype(Y_name_types)
            print("Testing on {} examples".format(np.sum( test_dates_ind ) ) )



# standard dataset (no sequencing)
class sent_Dataset(sent_Dataset_general):
    # subclassing of sent_Dataset
    # operator overloading for len and item
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self,
                    idx: int
                    ) -> Tuple[np.array , np.array, int  ]:
        return ( self.Xs[idx,...],self.Ys[ [idx],...],idx ) # (self.df.loc[idx, self.X_names].values, self.df.loc[idx, self.Y_names].values)


# sequence loading
class sent_Dataset_sequence(sent_Dataset_general):
    # subclassing of sent_Dataset: sequence version for RNNs
    # operator overloading for len and item
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self,
                    idx: int
                    ) -> Tuple[ np.array, np.array, int  ]:
        if idx >= self.sequence_length - 1:
            i_start = idx - self.sequence_length + 1
            Xseq = self.Xs[i_start:(idx + 1), :]
        else:
            # pad with 0s if seq exceeded
            zeropad = np.zeros([self.sequence_length-idx-1,self.Xs.shape[1]], dtype= self.Xs.dtype)
            Xseq = self.Xs[0:(idx + 1), :]
            Xseq = np.concatenate((zeropad,Xseq ))
        return (Xseq, self.Ys[idx], idx)



# the loader
def load_sent_data(
        df: pd.DataFrame,
        df_cutoff: datetime.datetime,
        train_batch_size: int,
        test_batch_size: int,
        X_names: List[str],
        Y_names: List[str],
        Y_name_types: Dict[str,Any ],
        train_dates: Union[pd.DatetimeIndex,None],
        convert_Y_to_labels: bool = True,
        shuffle: bool = True,
        kwargs: Union[Dict[str, Any], None] = {},
        datasetclass: Type[torch.utils.data.Dataset] = sent_Dataset,
        seq_len: int = 21,
        test_dates: Union[pd.DatetimeIndex,None] = None,
        lead_Y: int = 0,
        drop_last_in: bool = True
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,torch.utils.data.Dataset, torch.utils.data.Dataset  ]:
    '''
    :param df: input data frame containing X and Y
    :param df_cutoff: discard data below this index value (date)
    :param Y_names: names of ys
    :param Y_name_types: name types (int or np.float32)
    :param X_names: names of xs
    :param train_dates: specific dates for the training set
    :param test_dates: the specific dates for the test set
    :param convert_Y_to_labels: convert Y to 0/1 for classification or leave as is
    :param seq_len: sequence length of data items (for subsequent RNN use)
    :param lead_Y: lead or lag the Y (int )
    :param train_batch_size: batch size train
    :param test_batch_size: batch sz test
    :param shuffle: random draws T/F
    :param kwargs: additional stuff
    :param datasetclass: which class version to take (sequence based or std)
    :param seq_len: sequence length for setting up RNN loader
    :return: tuple of train loader, test loader, train data, test data
    '''
    #
    # more compact train / test loader call
    #
    train_data = datasetclass(df = df,df_cutoff = df_cutoff,
                     train=True, Y_names = Y_names,Y_name_types = Y_name_types, X_names = X_names,
                     train_dates=train_dates,convert_Y_to_labels = convert_Y_to_labels, seq_len = seq_len, test_dates=test_dates, lead_Y=lead_Y)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size, shuffle=shuffle, drop_last=drop_last_in,  **kwargs)

    test_data = datasetclass(df = df,df_cutoff = df_cutoff,
                     train=False,Y_names = Y_names,Y_name_types = Y_name_types, X_names = X_names,
                     train_dates=train_dates, convert_Y_to_labels = convert_Y_to_labels, seq_len = seq_len, test_dates=test_dates, lead_Y=lead_Y)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size, shuffle=shuffle, **kwargs)

    return train_loader, test_loader, train_data, test_data
