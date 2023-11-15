import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import re
from copy import deepcopy
from scipy import stats


# to wide format
def get_ESSnCSS_sent(dict, sentdf, countrylist, name, sentcol, timestampname,include_CSS = False, groupsum_cntrs = False,include_ESS_counts = True):
    '''
    :param dict:  dict to transform
    :param sentdf:  data frame with sents
    :param countrylist:  list of country denoms
    :param name:  slot in dict to be filled
    :param sentcol: sentiment column name
    :param timestampname:  name of time stamp
    :param include_CSS:   include CSS?
    :param groupsum_cntrs: group by country and sum up sentiments?
    :return:  transformed dict
    '''
    sentdf = sentdf[np.in1d(sentdf.COUNTRY_CODE, np.array([countrylist]))]
    if not groupsum_cntrs:
        # save countrycode in sentcol
        sentdf[sentcol] = sentdf.apply(lambda x: x.loc['COUNTRY_CODE'] + '_' + x.loc[sentcol],axis=1)  # append country code to group
        sentdf_ESS = deepcopy(sentdf[[timestampname, sentcol, 'ESS_MEAN', 'ESS_COUNT']])
    else:
        # group and sum up countries
        # "unmean"
        sentdf.loc[sentdf.ESS_COUNT>0,'ESS_MEAN']  = sentdf.loc[sentdf.ESS_COUNT>0,'ESS_MEAN']  * sentdf.loc[sentdf.ESS_COUNT>0,'ESS_COUNT']
        # group and sum
        sentdf_ESS = sentdf.groupby([timestampname,sentcol ]).agg( {'ESS_MEAN': np.nansum, 'ESS_COUNT': np.nansum} )
        sentdf_ESS.reset_index(inplace=True)
        # get means again
        sentdf_ESS.loc[sentdf_ESS.ESS_COUNT>0,'ESS_MEAN'] = sentdf_ESS.loc[sentdf_ESS.ESS_COUNT>0,'ESS_MEAN'] / sentdf_ESS.loc[sentdf_ESS.ESS_COUNT>0,'ESS_COUNT']
    #
    sentdf_ESS = sentdf_ESS[sentdf_ESS.ESS_COUNT > 0]
    # a) sentiment:
    # pivot
    sentdf_ESS_sent = sentdf_ESS.pivot_table(index=[timestampname], columns=[sentcol], values=['ESS_MEAN'], fill_value=0.0)
    # index to time
    sentdf_ESS_sent.index = pd.Index(pd.to_datetime(sentdf_ESS_sent.index, format='%Y-%m-%d'), name="DATE")
    # flatten & rename cols
    sentdf_ESS_sent.columns = [
        sentdf_ESS_sent.columns.get_level_values(1)[i] + '_' + sentdf_ESS_sent.columns.get_level_values(0)[i].split('_')[0]+'_'+sentcol for i
        in range(0, len(sentdf_ESS_sent.columns))]
    # b) counts
    # pivot
    sentdf_ESS_cnt = sentdf_ESS.pivot_table(index=[timestampname], columns=[sentcol], values=['ESS_COUNT'],
                                             fill_value=0.0)
    # index to time
    sentdf_ESS_cnt.index = pd.Index(pd.to_datetime(sentdf_ESS_cnt.index, format='%Y-%m-%d'), name="DATE")
    # flatten & rename cols
    sentdf_ESS_cnt.columns = [
        sentdf_ESS_cnt.columns.get_level_values(1)[i] + '_count' +
        sentdf_ESS_cnt.columns.get_level_values(0)[i].split('_')[0] for i
        in range(0, len(sentdf_ESS_cnt.columns))]
    # log of counts
    for c in sentdf_ESS_cnt.columns:
        sentdf_ESS_cnt[c] = np.log(sentdf_ESS_cnt[c]+1.0)

    if include_ESS_counts:
        # merge count and ESS
        sentdf_ESS = sentdf_ESS_sent.merge(sentdf_ESS_cnt, on=['DATE'], how='left')
    else:
        sentdf_ESS = sentdf_ESS_sent

    if include_CSS:
        if not groupsum_cntrs:
            sentdf_CSS = deepcopy( sentdf[[timestampname, sentcol, 'CSS_MEAN', 'CSS_COUNT']] )
        else:
            # group and sum up countries
            # "unmean"
            sentdf.loc[sentdf.CSS_COUNT > 0, 'CSS_MEAN'] = sentdf.loc[sentdf.CSS_COUNT > 0, 'CSS_MEAN'] * sentdf.loc[
                sentdf.CSS_COUNT > 0, 'CSS_COUNT']
            # group and sum
            sentdf_CSS = sentdf.groupby([timestampname, sentcol]).agg({'CSS_MEAN': np.nansum, 'CSS_COUNT': np.nansum})
            sentdf_CSS.reset_index(inplace=True)
            # get means again
            sentdf_CSS.loc[sentdf_CSS.CSS_COUNT > 0, 'CSS_MEAN'] = sentdf_CSS.loc[sentdf_CSS.CSS_COUNT > 0, 'CSS_MEAN'] / sentdf_CSS.loc[
                sentdf_CSS.CSS_COUNT > 0, 'CSS_COUNT']
        sentdf_CSS = sentdf_CSS[sentdf_CSS.CSS_COUNT > 0]
        sentdf_CSS = sentdf_CSS.pivot_table(index=[timestampname], columns=[sentcol], values=['CSS_MEAN'],fill_value=0.0)
        sentdf_CSS.index = pd.Index(pd.to_datetime(sentdf_CSS.index, format='%Y-%m-%d'), name="DATE")
        sentdf_CSS.columns = [
            sentdf_CSS.columns.get_level_values(1)[i] + '_' + sentdf_CSS.columns.get_level_values(0)[i].split('_')[0] +'_'+sentcol
            for i in range(0, len(sentdf_CSS.columns))]
        # merge ESS on CSS (left) / CSS always filled
        sentdf = sentdf_CSS.merge(sentdf_ESS, on=['DATE'], how='left')  # index name is "Date"
        dict[name] = sentdf
    else:
        dict[name] = sentdf_ESS
    return dict


# chunker
def chunks(l, n):
    '''
    :param l: list
    :param n: chunk size
    :return: generator, gen next chunk. yields successive n-sized chunks from l
    '''
    for i in range(0, len(l), n):
        yield list(l[i:i + n])


# plotting
def plot_all_sent_ts(dict,plotting_cutoff, pdfname_suffix, apply_rolling ):
    '''
    :param dict: dictionary of sentiment dataframes
    :param plotting_cutoff:  datetime cutoff
    :param senttype:  'group' or 'type'
    :return: None
    '''
    for n in dict.keys():
        print('plot all variables for '+ n)
        sent = dict[n]
        if apply_rolling:
            for c in sent.columns:
                sent[c] = sent[c].rolling(window=21*3, min_periods=21).mean()
        # out to pdf
        pp = PdfPages(n + pdfname_suffix + '.pdf')
        for c in chunks(range(0, sent.shape[1]), 4):
            f = plt.figure()
            for plotnr, i in enumerate(c):
                plt.subplot(2, 2, plotnr + 1)
                sent.loc[sent.index > plotting_cutoff, sent.columns[i]].plot(title=sent.columns[i])
            pp.savefig(f)
            plt.close()
        pp.close()



def extract_macro_Y( midas_Y, macroseries, justsetindex = False ):
    '''
    :param midas_Y: dict container for Y extracted from the macro data
    :param macroseries: dict of macrodata
    :return: None, modifies dict
    '''
    for n in macroseries.keys():
        # change index
        #macroseries[n].set_index(pd.Index(pd.to_datetime([x.date() for x in macroseries[n].index._data]), name='DATE'), inplace=True)
        macroseries[n].set_index(pd.Index(pd.to_datetime(macroseries[n].index), name='DATE'),inplace=True)
        if n != 'CN' and not justsetindex:
            # add macro
            midas_Y[n] = macroseries[n]['orig_NonVintage_GDP_QoQ_growth_SA']
            # then drop
            macroseries[n].drop(['orig_NonVintage_GDP_QoQ_growth_SA'], inplace=True, axis = 1)
            # drop nans
            midas_Y[n] = midas_Y[n].loc[np.logical_not(np.isnan(midas_Y[n]))]




def merge_macro_n_sent(macroseries, sentdfs_group):
    '''
    :param macroseries:  ECB + OECD macro data
    :param sentdfs_group:  Ravenpack sent, group aggregation
    :param sentdfs_type:   Ravenpack sent, type agg
    :return:  changes input dictionary slots
    '''
    # merge
    for n in macroseries.keys():
        print(' join for ' + n)
        macroseries[n].rename(columns=lambda x: 'M_' + x, inplace=True)
        # transform index
        macroseries[n]['indexcol'] = macroseries[n].index
        # need day monthy year format:
        macroseries[n]['indexcol'] = macroseries[n].apply(lambda x: np.datetime64(x.loc['indexcol'].date()), axis=1)
        macroseries[n]['indexcol'] = pd.to_datetime(macroseries[n]['indexcol'], format='%Y-%m-%d')
        macroseries[n].set_index(pd.Index(macroseries[n]['indexcol'], name='DATE'), inplace=True)
        macroseries[n].drop(['indexcol'], axis=1, inplace=True)
        # merge to sentdfs
        sentdfs_group[n] = sentdfs_group[n].merge(macroseries[n],on = ['DATE'], how = 'left' )
        # merge to sentdfs
        #sentdfs_type[n] = sentdfs_type[n].merge(macroseries[n], on=['DATE'], how='left')
        # fill values of macroseries
        for cname in macroseries[n].columns:
            sentdfs_group[n][cname].fillna(method = 'ffill' , inplace = True)
            #sentdfs_type[n][cname].fillna(method='ffill', inplace=True)


def get_GDELTTone_tables(sent_gdelt, cntrs ):
    '''
    :param sent_gdelt:  GDELT data (inc TONE sent)
    :param cntrs:   countries to aggregate
    :return:   tuple of Tone, PosTone, and NegTone pandas
    '''
    sentdf_GTone = deepcopy(sent_gdelt)
    sentdf_GTone.index = pd.Index(pd.to_datetime(sentdf_GTone.TIMESTAMP_UTC, format='%Y-%m-%d'), name="DATE")
    #  sentiment: pool sents
    sentdf_GTone = sentdf_GTone[np.in1d(sentdf_GTone.COUNTRY_CODE, np.array([cntrs]))]
    # group by  date/ topic  and combine -> sum and combine
    # careful: Ravenpack: have MEANS and COUNTS. GDELT: have SUMS and COUNTS
    sentdf_GTone = sentdf_GTone.groupby(['DATE', 'ENTITY_TYPE' ]).agg({'AVG_TONE_SCORE': np.nansum , 'AVG_TONE_COUNT': np.nansum,
                                                        'POS_TONE_SCORE': np.nansum , 'NEG_TONE_SCORE':np.nansum }) # compare to apply!
    sentdf_GTone.reset_index(inplace = True)
    # get means
    sentdf_GTone['AVG_TONE_SCORE'] = sentdf_GTone['AVG_TONE_SCORE'] / sentdf_GTone['AVG_TONE_COUNT']
    sentdf_GTone['POS_TONE_SCORE'] = sentdf_GTone['POS_TONE_SCORE'] / sentdf_GTone['AVG_TONE_COUNT']
    sentdf_GTone['NEG_TONE_SCORE'] = sentdf_GTone['NEG_TONE_SCORE'] / sentdf_GTone['AVG_TONE_COUNT']
    # now pivot out three tables:
    sentdf_GPos = sentdf_GTone.pivot_table(index=['DATE'], columns=['ENTITY_TYPE'], values=['POS_TONE_SCORE'], fill_value=np.nan)
    sentdf_GNeg = sentdf_GTone.pivot_table(index=['DATE'], columns=['ENTITY_TYPE'], values=['NEG_TONE_SCORE'], fill_value=np.nan)
    sentdf_GTone = sentdf_GTone.pivot_table(index=['DATE'], columns=['ENTITY_TYPE'], values=['AVG_TONE_SCORE'], fill_value=0.0)#
    # rename cols
    sentdf_GPos.columns = sentdf_GPos.columns.get_level_values(1)
    sentdf_GNeg.columns = sentdf_GNeg.columns.get_level_values(1)
    sentdf_GTone.columns = sentdf_GTone.columns.get_level_values(1)

    return sentdf_GTone, sentdf_GPos, sentdf_GNeg



def get_GDELTGCAM_tables(sent_gdelt, cntrs, scorename ):
    '''
    :param sent_gdelt:  GDELT data (inc TONE sent)
    :param cntrs:   countries to aggregate
    :param scorename:  specify lexicon (sent score)
    :return:   tuple of Tone, PosTone, and NegTone pandas
    '''
    sentdf_GCAM = deepcopy(sent_gdelt)
    sentdf_GCAM.index = pd.Index(pd.to_datetime(sentdf_GCAM.TIMESTAMP_UTC, format='%Y-%m-%d'), name="DATE")
    #  sentiment: pool sents
    sentdf_GCAM = sentdf_GCAM[np.in1d(sentdf_GCAM.COUNTRY_CODE, np.array([cntrs]))]
    # group by  date/ topic  and combine -> sum and combine
    #
    # "unmean"
    sentdf_GCAM[scorename] = sentdf_GCAM[scorename] * sentdf_GCAM['LMCD_COUNT']
    # group and sum
    sentdf_GCAM = sentdf_GCAM.groupby(['DATE', 'ENTITY_TYPE' ]).agg({scorename: np.nansum , 'LMCD_COUNT': np.nansum }) # compare to apply!
    sentdf_GCAM.reset_index(inplace = True)
    # get means again
    sentdf_GCAM[scorename] = sentdf_GCAM[scorename] / sentdf_GCAM['LMCD_COUNT']
    # now pivot out counts and sentiments
    sentdf_GCAM_sent = sentdf_GCAM.pivot_table(index=['DATE'], columns=['ENTITY_TYPE'], values=[scorename], fill_value=0.0)#
    # log of count
    sentdf_GCAM['LMCD_COUNT'] = np.log(sentdf_GCAM['LMCD_COUNT']+1.0)
    sentdf_GCAM_cnt = sentdf_GCAM.pivot_table(index=['DATE'], columns=['ENTITY_TYPE'], values=['LMCD_COUNT'],fill_value=0.0)  #
    # rename cols
    sentdf_GCAM_sent.columns = sentdf_GCAM_sent.columns.get_level_values(1)
    sentdf_GCAM_cnt.columns = sentdf_GCAM_cnt.columns.get_level_values(1)

    return sentdf_GCAM_sent, sentdf_GCAM_cnt



def merge_DATAtoRP(sentdfs_left, sentdfs_right, fill_method = None, fill_value = None, change_right_colnames = False, right_prefix = 'TYPE_'):
    '''
    :param sentdfs_left: dict of pandas dfs
    :param sentdfs_right:  dict of pandas dfs
    :return: None (modifies input dicts)
    '''
    # merge selected GDELT dicts and macro data to Ravenpack ...
    for n in sentdfs_left.keys():
        if n in sentdfs_right.keys():
            if change_right_colnames:
                sentdfs_right[n].rename(columns=lambda x: right_prefix + x, inplace=True)
            # merge to sentdfs
            sentdfs_left[n] = sentdfs_left[n].merge(sentdfs_right[n],on = ['DATE'], how = 'left' )
            # fill values
            if type(sentdfs_right[n]) == pd.core.frame.DataFrame:
                loopover = sentdfs_right[n].columns
            else:
                loopover = [sentdfs_right[n].name]
            for cname in loopover:
                if fill_value is not None:
                    sentdfs_left[n][cname].fillna(fill_value, inplace=True)

                if fill_method is not None:
                    sentdfs_left[n][cname].fillna(method=fill_method, inplace=True)



# rolling standardization / demeaning
def zscore(x, window, min_periods = 10, lag = 0, demean = True, stdze = True, min_std = 0.0001, expanding_switch = False, apply_sigmoid = False,
           annualize_to_target = None, return_scale_only = False):
    '''
    :param x:  pd.series
    :param window: as in pd...rolling
    :param min_periods:  same as in pd..rolling
    :param lag: lag std and mean?
    :param demean: apply demeaning
    :param stdze: apply standardization
    :param min_std: minimum std dev
    :param expanding_switch: expanding or rolling?
    :param for signals, apply sigmoid in the end?
    :param annualize_to_target: float ann. target vola or None if not applicable
    :param return_scale_only: T/F -> return vola scale only if T ( 1/vol )
    :return: z-scored series
    '''
    if not expanding_switch:
        r = x.rolling(window=window,min_periods = min_periods)
    else:
        r = x.expanding( min_periods=min_periods )
    m, s = 0.0, 1.0
    if demean:
        m = r.mean().shift(lag)
    if stdze:
        s = r.std(ddof=0).shift(lag)
        s[s<=min_std] = min_std

    if annualize_to_target is None:
        z = (x-m)/s
    else:
        z = (x-m) / (s*np.sqrt(252)) * annualize_to_target
    # overwrite z if only the scale of interest
    if return_scale_only:
        z = 1.0 / (s * np.sqrt(252)) * annualize_to_target
        # df = df / (df.std() * np.sqrt(252)) * targetvol_ann
    # if sigmoid needed
    if apply_sigmoid:
        z = 1/(1+np.exp(-z))
    return z


def zscore_this_dict(sent_dict,
                     window_in,
                     min_per_in,
                     demean_in = True,
                     stdze_in = True,
                     min_std_in = 0.001,
                     nan_handling = False, nan_handling_method = 'ffill',
                     nan_fill_value = None,
                     expanding_switch_in =False, apply_sigmoid = False,
                     lag = 0, annualize_to_target = None, return_scale_only = False):
    '''
    :param sent_dict: dict of sentiments as before
    :param window_in: if rolling, window size
    :param min_per_in: minimum periods in expanding and rolling windows
    :param demean_in: demean it ?
    :param stdze_in:  standardize it?
    :param min_std_in:  minimum std (cutoff)
    :param nan_handling: fillna?
    :param nan_handling_method:  fillna method
    :param nan_fill_value: how to handle empty entries -> fill with this
    :param expanding_switch_in: expanding or rolling
    :param apply_sigmoid: sigmoid function after std + demeaning?
    :param lag: lag standardization, i.e. use historical vola and not current obs
    :param annualize_to_target: float ann. target vola or None if not applicable
    :param return_scale_only: True / False if only scaling is of interest
    :return:  None / modifies sent dict
    '''
    # zscore of sentiment series
    for n in sent_dict.keys():
        if nan_handling:
            if nan_fill_value is None:
                sent_dict[n] = sent_dict[n].fillna(method= nan_handling_method, axis = 0)
            else:
                sent_dict[n] = sent_dict[n].fillna(nan_fill_value)
        for c in sent_dict[n].columns:
            sent_dict[n][c] = zscore(sent_dict[n][c], window=window_in, min_periods=min_per_in, demean=demean_in,
                                     stdze=stdze_in, min_std=min_std_in, expanding_switch = expanding_switch_in, apply_sigmoid = apply_sigmoid, lag=lag,
                                     annualize_to_target = annualize_to_target, return_scale_only = return_scale_only)



def bring_to_same_index( index_provider_dict, right_dict, on_index_name_list= ['DATE'], how_in = 'left', nan_handling = False, fillmethod = 'ffill',
                         nan_fill_value = None):
    '''
    :param index_provider_dict: not modified, provides index
    :param right_dict: dict of pd obj to be modified ( these are supposed to be merged to the static_dict index)
    :param on_index_name_list: on -> index name to merge on
    :param how_in: type of merge
    :param nan_handling: handle nans
    :param fillmethod: how to handle nans
    :param nan_fill_value: how to handle empty entries -> fill with this
    :return: out_dict, same format as right_dict
    '''
    # each merge pair with mutually exclusive col names and same index name
    out_dict = defaultdict()
    for n in right_dict.keys():
        print(n)
        mergeonthis = pd.DataFrame( data = {'test': np.zeros(shape = (index_provider_dict[n].shape[0],))},
                                    index = pd.Index(index_provider_dict[n].index._data, name = on_index_name_list[0] ) )
        # merge
        out_dict[n] = mergeonthis.merge(right_dict[n], on=on_index_name_list, how= how_in)
        # remove cols
        out_dict[n].drop(['test'], inplace = True, axis = 1)
        if nan_handling:
            if nan_fill_value is None:
                out_dict[n] = out_dict[n].fillna(method=fillmethod, axis = 0)
            else:
                out_dict[n] = out_dict[n].fillna(nan_fill_value)
    return out_dict



def get_long_term_rets( returns, dayslist = {'wk': 5, 'mth': 21, 'sixm': 21*6, 'yr': 21*12}, aggtype = 'sum',
                        get_change = False, get_change_list = {'mth': 21}):
    '''
    :param returns: pandas with idx returns (wide format)
    :param dayslist: dict of names (keys will be added to col names)
    :param idx_country_map: how to distribute idxs to countries
    :return: idx return dict
    '''
    for n in returns.keys():
        for c in returns[n].columns:
            # get
            if not get_change:
                for t,d in dayslist.items():
                    #
                    if aggtype == 'sum':
                        returns[n][c+'_'+t] = returns[n][c].rolling(window = d, min_periods = d ).sum()
                    elif aggtype == 'mean':
                        returns[n][c + '_' + t] = returns[n][c].rolling(window=d, min_periods=d).mean()
            if get_change:
                for t_chg,d_chg in get_change_list.items():
                    returns[n][c + '_' + t_chg+ '_'+ str(d_chg)+ 'chg'] = returns[n][c].diff(periods = d_chg)

    return returns


def zeropad( sentdf_dict, fill_v = 0.0 ):
    for n in sentdf_dict.keys():
        sentdf_dict[n] = sentdf_dict[n].fillna( value = fill_v )
    return sentdf_dict

def get_colname_dicts( data_dict, dataset_str_patterns = {'gdelt' :  'GC_(.*?)',
                                                        'CSS_GROUP' : '(.*?)_CSS_GROUP',
                                                        'ESS_GROUP' : '(.*?)_ESS_GROUP',
                                                        'CSS' : '(.*?)_CSS_(.*?)',
                                                        'ESS' : '(.*?)_ESS_(.*?)',
                                                        'macro' : 'M_(.*?)', 'zylds': 'ZYLD_(.*?)','eqidx': 'EQidx_(.*?)',
                                                        'eqidxvol': 'EQidxvol_(.*?)','fx': 'FX_(.*?)',
                                                        'ir': 'IR3M_(.*?)', 'RPcnt': '(.*?)_countESS','Gcnt': 'GCcnt_(.*?)', 'cmdty': 'c_(.*?)',
                                                        'CSS_TYPE' : '(.*?)_CSS_TYPE','ESS_TYPE' : '(.*?)_ESS_TYPE',
                                                        'sentPCA': 'SENT_(.*?)'} ):
    '''
    :param data_dict:   dict containing pandas dfs per country
    :param dataset_str_patterns:  substr patterns indicating data set
    :return: dict containing dataset keys and col names for these datasets
    '''
    colname_dict = defaultdict()
    for n in data_dict.keys():
        dataset_col_dict = defaultdict()
        for d in dataset_str_patterns.keys():
            dataset_col_dict[d] = [ x for x in  data_dict[n].columns if re.match( dataset_str_patterns[d], x)]
        colname_dict[n] = dataset_col_dict

    return colname_dict



def distribute_wide_to_dict(widepd, addprefix, map = {'AUD':'AU','CAD':'CA','GER':'EU','GBP':'GB','JPY':'JP','USD':'US'}):
    '''
    :param widepd: pandas time series wide format
    :param map: col name map
    :return: dict according to map
    '''
    if 'DATE' in widepd.columns:
        widepd.index = pd.Index(pd.to_datetime(widepd['DATE'], format='%Y-%m-%d'), name="DATE")
        widepd.drop(['DATE'], axis=1, inplace=True)
    dict_out = defaultdict()
    for n in widepd.columns:
        # map
        dict_out[map[n]] = deepcopy( widepd[n] ).to_frame()
        dict_out[map[n]].columns = [addprefix + n ]

    return dict_out



def extract_data_by_colname(dict_in, colname_dict, subdataname, dropcols_in = True ):
    '''
    :param dict_in: dict of sent nr of news combination etc
    :param colname_dict: dict of colnames for different data sets
    :param subdataname: data set name
    :param dropcols_in:  drop cols in dict_in after moving to out_dict?
    :return: dictionary containing extracted dataset. keys are countries
    '''
    out_dict = defaultdict()
    for n in dict_in.keys():
        out_dict[n] = dict_in[n][colname_dict[n][subdataname]]
        # drop
        if dropcols_in:
            dict_in[n].drop( colname_dict[n][subdataname], axis = 1, inplace = True)
    return out_dict



def get_cmdty(cmdty_CRB, cmdty_BBG, sentdfs, cutdate):
    '''
    :param cmdty_CRB, cmdty_BBG:  commodity index
    :param sentdfs:  Ravenpack sent, group aggregation
    :param cutdate:  date cutoff
    :return:  standard dict with region  / country keys
    '''

    cmdty_BBG.set_index(pd.Index(pd.to_datetime(cmdty_BBG['Index'], format='%Y-%m-%d'), name="DATE"), inplace=True);
    cmdty_BBG.drop(['Index'], axis=1, inplace=True)
    cmdty_CRB.set_index(pd.Index(pd.to_datetime(cmdty_CRB.Dates, format='%d.%m.%Y'), name="DATE"), inplace=True);
    cmdty_CRB.drop(['Dates'], axis=1, inplace=True)
    cmdty_CRB.rename({'CRYTR Index': 'c_CRBcmdtyRet','DXY Curncy':'FX_DXYcrncyRet'}, axis=1, inplace=True)
    cmdty_BBG.rename({'Price': 'c_BBGcmdtyRet'}, axis=1, inplace=True)
    # merge both TS
    cmdty_CRB = cmdty_CRB.merge(cmdty_BBG, on=['DATE'], how='left')
    cmdty_CRB = cmdty_CRB.loc[cmdty_CRB.index>=cutdate]
    cmdty_CRB = cmdty_CRB.fillna(method = 'ffill')
    cmdty_CRB = cmdty_CRB.fillna(method = 'bfill') # kill leading NaNs in BBG index (not used anyway)

    cmdty = defaultdict()

    # merge
    for n in sentdfs.keys():
        cmdty[n] = deepcopy(cmdty_CRB)
        # get returns
        cmdty[n] = cmdty[n].pct_change()

        # merge to sentdfs
        #sentdfs[n] = sentdfs[n].merge(cmdty_BBG,on = ['DATE'], how = 'left' )
        # merge to sentdfs
        #sentdfs[n] = sentdfs[n].merge(cmdty_CRB, on=['DATE'], how='left')
        # fill values
        #sentdfs[n]['c_CRBcmdtyIdx'].fillna(method = 'ffill' , inplace = True)
        #sentdfs[n][ 'c_BBGcmdtyIdx' ].fillna(method='ffill', inplace=True)

    return cmdty



def hausman(fe, re):
    '''
    :param fe: fixed effects estimation obj from statsmodels
    :param re: random effects estimation obj from statsmodels
    :return: tuple (test value, degrees of freedom, p value)
    '''
    # hausman test from https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
    b_fe = fe.params
    b_re = re.params
    v_b_fe = fe.cov
    v_b_re = re.cov
    df = b_fe[np.abs(b_fe) < 1e8].size
    chi2 = np.dot( (b_fe - b_re).T, np.linalg.inv( v_b_fe-v_b_re ).dot(b_fe-b_re) )
    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval



def get_EMA_Xover(signal,
                  EMA_specs={'fast': [20], 'slow': [50], 'wgt': [1.0]},
                  EMA_switch=False,
                  EMA_colname_suffix='_EMA', convert_to_binary = False):
    '''
    :param signal: data frame input
    :param EMA_specs: slots fast slow and weights. slots can be empty
    :param EMA_switch: EMA or MA
    :param EMA_colname_suffix: name suffix for new col
    :return: modified signal df
    '''
    # create crossover cols
    signalnames = signal.columns
    for n in signal.columns:
        signal[n + EMA_colname_suffix] = 0.0
        if not EMA_switch:
            # fast MA - slow MA
            for i in range(0, len(EMA_specs['slow'])):
                signal[n + EMA_colname_suffix] -= EMA_specs['wgt'][i] * (
                    signal[n].rolling(window=EMA_specs['slow'][i], min_periods=1).mean())
            for i in range(0, len(EMA_specs['fast'])):
                signal[n + EMA_colname_suffix] += EMA_specs['wgt'][i] * (
                    signal[n].rolling(window=EMA_specs['fast'][i], min_periods=1).mean())
        else:
            # fast EMA - slow EMA
            for i in range(0, len(EMA_specs['slow'])):
                signal[n + EMA_colname_suffix] -= EMA_specs['wgt'][i] * (
                    signal[n].ewm(halflife=EMA_specs['slow'][i], adjust=False).mean())
            for i in range(0, len(EMA_specs['fast'])):
                signal[n + EMA_colname_suffix] += EMA_specs['wgt'][i] * (
                    signal[n].ewm(halflife=EMA_specs['fast'][i], adjust=False).mean())

        if convert_to_binary:
            signal[n + EMA_colname_suffix] = (signal[n + EMA_colname_suffix]>=0)*1 - (signal[n + EMA_colname_suffix]<0)*1

    return signal[ [n + EMA_colname_suffix  for n in signalnames] ]



class bt:
    def __init__(self): pass
    def fill_rebal_weights(self): pass
    def calc_returns(self): pass
    def change_expostvola(self): pass