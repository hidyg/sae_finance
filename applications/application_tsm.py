import os
import pandas as pd
import pickle as pckl
import applications.BTdata_utils as BTdata_utils
import datetime
import numpy as np
from copy import deepcopy
from applications.BTdata_utils import bt
import statsmodels.api as sm
import itertools


#if __name__=='__main__':
cutoff = datetime.datetime.strptime('2004-01-01', '%Y-%m-%d')

# load sentiment representations
with open('data/workspace/sent_NNresults_test_semiannual_retrain_lead0m.pickle','rb') as output_file:
    sentdfs = pckl.load(output_file)

################################################
# TSM strats inc. plotting
################################################
# a) need vola-scaled returns (EQ, FI, FX)
# add fxf returns
# FX Forwards: 16:00 GMT
fxfrets = pd.read_csv('./data/workspace/FXFreturns.csv', index_col= 0)
fxfrets_dict_inst = BTdata_utils.distribute_wide_to_dict(fxfrets, 'FXFi_', map = {'AUD':'AU','CAD':'CA','EUR':'EU','GBP':'GB','JPY':'JP','USD':'US'})
fxfrets_dict_signal = BTdata_utils.distribute_wide_to_dict(fxfrets, 'FXFs_', map = {'AUD':'AU','CAD':'CA','EUR':'EU','GBP':'GB','JPY':'JP','USD':'US'})
fxfrets_dict_V = BTdata_utils.distribute_wide_to_dict(fxfrets, 'FXFv_', map = {'AUD':'AU','CAD':'CA','EUR':'EU','GBP':'GB','JPY':'JP','USD':'US'})


# add fi futures returns
firets = pd.read_csv('./data/workspace/FIreturns.csv', index_col=0 )
firets_dict_inst = BTdata_utils.distribute_wide_to_dict(firets, 'FIFi_', map = {'Bund':'EU', 'US10YrNote':'US','LongGilt':'GB','Canada10Yr':'CA','Australia10Yr':'AU',
       'Japan10Yr':'JP'})
firets_dict_signal = BTdata_utils.distribute_wide_to_dict(firets, 'FIFs_', map = {'Bund':'EU', 'US10YrNote':'US','LongGilt':'GB','Canada10Yr':'CA','Australia10Yr':'AU',
       'Japan10Yr':'JP'})
firets_dict_V = BTdata_utils.distribute_wide_to_dict(firets, 'FIFv_', map = {'Bund':'EU', 'US10YrNote':'US','LongGilt':'GB','Canada10Yr':'CA','Australia10Yr':'AU',
       'Japan10Yr':'JP'})


# add eq returns
eqfutrets = pd.read_csv('./data/workspace/EQfutreturns.csv')
eqfutrets_dict_inst = BTdata_utils.distribute_wide_to_dict(eqfutrets, 'EQFi_',map = {'EUROSTOXX50': 'EU', 'FTSE100':'GB', 'HangSeng':'CN', 'S&P/TSE60Canada':'CA',
                                                                                'S&P500Mini':'US', 'SPI200': 'AU', 'TOPIX': 'JP'  } )
eqfutrets_dict_signal = BTdata_utils.distribute_wide_to_dict(eqfutrets, 'EQFs_',map = {'EUROSTOXX50': 'EU', 'FTSE100':'GB', 'HangSeng':'CN', 'S&P/TSE60Canada':'CA',
                                                                                'S&P500Mini':'US', 'SPI200': 'AU', 'TOPIX': 'JP'  } )
eqfutrets_dict_V = BTdata_utils.distribute_wide_to_dict(eqfutrets, 'EQFv_',map = {'EUROSTOXX50': 'EU', 'FTSE100':'GB', 'HangSeng':'CN', 'S&P/TSE60Canada':'CA',
                                                                                'S&P500Mini':'US', 'SPI200': 'AU', 'TOPIX': 'JP'  } )

bring_to_same_index = True
if bring_to_same_index:
    # FX and FI returns to same index as EQ:
    firets_dict_inst = BTdata_utils.bring_to_same_index( eqfutrets_dict_inst, firets_dict_inst, on_index_name_list= ['DATE'],
                                            how_in = 'left', nan_handling = True, fillmethod = 'None',nan_fill_value = 0.0)
    fxfrets_dict_inst = BTdata_utils.bring_to_same_index( eqfutrets_dict_inst, fxfrets_dict_inst, on_index_name_list= ['DATE'],
                                            how_in = 'left', nan_handling = True, fillmethod = 'None',nan_fill_value = 0.0)



# vola scaling / z-scoring
# -> extract
# -> check implementation
BTdata_utils.zscore_this_dict( eqfutrets_dict_V, window_in = int(252)*2, min_per_in= int(252)*2,nan_handling=True, expanding_switch_in = False,
                               min_std_in = 0.001 , stdze_in = True, demean_in= False, lag=0, annualize_to_target= 0.01, return_scale_only=True )
BTdata_utils.zscore_this_dict( firets_dict_V,window_in = int(252)*2, min_per_in= int(252)*2, nan_handling=True, expanding_switch_in = False,
                               min_std_in = 0.001 , stdze_in = True, demean_in= False, lag=0, annualize_to_target= 0.01, return_scale_only=True )
BTdata_utils.zscore_this_dict( fxfrets_dict_V,window_in = int(252)*2, min_per_in= int(252)*2,nan_handling=True, expanding_switch_in = False,
                               min_std_in = 0.001 ,  stdze_in = True, demean_in= False, lag=0, annualize_to_target= 0.01, return_scale_only=True )



# merge to sentdfs dict, fill-value 0
BTdata_utils.merge_DATAtoRP(sentdfs,  fxfrets_dict_signal, fill_value= 0.0, fill_method= None)
BTdata_utils.merge_DATAtoRP(sentdfs,  firets_dict_signal, fill_value= 0.0, fill_method= None)
BTdata_utils.merge_DATAtoRP(sentdfs,  eqfutrets_dict_signal, fill_value= 0.0, fill_method= None)
BTdata_utils.merge_DATAtoRP(sentdfs,  fxfrets_dict_V, fill_value= None, fill_method= 'ffill')
BTdata_utils.merge_DATAtoRP(sentdfs,  firets_dict_V, fill_value= None, fill_method= 'ffill')
BTdata_utils.merge_DATAtoRP(sentdfs,  eqfutrets_dict_V, fill_value= None, fill_method= 'ffill')



# set up dictionaries
NN_prefix =  'GRUAE_d_' # 'GRU_d_'
# for GRU / GRUAE, need one more var:
sentdata_dict = BTdata_utils.get_colname_dicts(sentdfs, dataset_str_patterns={'EQidx': 'EQidx_(.*?)','FXFs': 'FXFs_(.*?)','EQidxvol': 'EQidxvol_(.*?)',
                                                                              'EQFs': 'EQFs_(.*?)', 'FIFs': 'FIFs_(.*?)',
                                                                              'c_CRBcmdtyRet': 'c_CRBcmdtyRet_(.*?)',
                                                                             'cmdty':'c_(.*?)',
                                                                               'NNeq': NN_prefix+'d_HEQidx_S&P500Mini',
                                                                               'NNfi': NN_prefix+ 'd_HZYLD_USD_d_1chg',
                                                                               'NNmacro':  NN_prefix+ 'd_HM_PC0_mth_30chg',
                                                                               'NNcmdty': NN_prefix+ 'd_Hc_CRBcmdtyRet',
                                                                               'NNfx':  NN_prefix+'d_HFX_DXYcrncyRet',
                                                                               'NN': NN_prefix + '(.*?)',
                                                                               'ZYLD': 'ZYLD_(.*?)', 'FX': 'FX_(.*?)'
                                                                              })


# Xovers specs
EMA = { 'fast': [21], 'slow': [], 'wgt': [1.0] }  #
# signal lags
signal_lag = 1
# get all strategy components per country
cntrs = {'EQ': list(sentdfs.keys()), 'FI': [x for x in list(sentdfs.keys()) if x != 'CN'],
         'FX': [x for x in list(sentdfs.keys()) if x not in ['CN','US'] ]}

BMs = {'EQ':'EQidx', 'FI': 'ZYLD', 'FX': 'FX'}
NN_slots = {'EQ': 0, 'FI': 1, 'FX': 4}


# EMAs to look at
EMA_list = [1,2,3,4,5, 10,  20,30,40,50,60,70,80,90  ]

reshuffling = 'daily' # 1st paper version: biweekly

# organize the volatility scaling
vol_scales = {'EQ': eqfutrets_dict_V, 'FI': firets_dict_V, 'FX': fxfrets_dict_V}
# organize the instrument returns
instr_rets = {'EQ': eqfutrets_dict_inst, 'FI': firets_dict_inst, 'FX': fxfrets_dict_inst}
global_colnames = [ 'Fs', 'NN',  'idx', 'Passive']


strats = [['EQ','FI','FX'] , ['EQ'], ['FI'], ['FX'] ]  # combined asset classes, individual
bts_strat = {}
save_res = True
save_res_all = True

for strat_included in strats:

    bts_all = {}


    for fastsignal in EMA_list:

        if not save_res:
            print('results will not be saved')
        bt_dfs = []
        EMA['fast'] = [fastsignal]
        print('iteration: '+str(fastsignal) )
        for strat in strat_included:#,'FI','FX']:

            print('current strat: '+str(strat_included))
            for cntr in cntrs[strat]:

                strat_components = {strat:[sentdata_dict[cntr][strat+'Fs'][0],sentdata_dict[cntr]['NN'][NN_slots[strat]],
                                           sentdata_dict[cntr][BMs[strat]][0] ]}#,
                                    #'FI':[sentdata_dict[cntr][strat+'Fs'][0],sentdata_dict[cntr]['NN'][1],sentdata_dict[cntr]['ZYLD'][0] ],
                                    #'FX':[sentdata_dict[cntr][strat+'Fs'][0],sentdata_dict[cntr]['NN'][4],sentdata_dict[cntr]['FX'][0] ]}

                # column mapper
                colmapper = dict(zip([x + '_EMA' for x in strat_components[strat]] + list(vol_scales[strat][cntr].columns),
                                   global_colnames))

                # get data for slot
                df = sentdfs[cntr][ strat_components[strat] ]
                df = df.loc[ df.index >= cutoff ]

                # compute MA (X-overs) ... these are the weights already if no sigmoid or else is used
                df_ema =  BTdata_utils.get_EMA_Xover(df, EMA_specs = EMA, EMA_switch=True, convert_to_binary = True )

                # each ema col is a different signal...
                df_ema = df_ema.merge( vol_scales[strat][cntr], on=['DATE'] ).fillna( method = 'ffill' ) # inner
                for i,c in enumerate(df_ema.columns):
                    if c not in vol_scales[strat][cntr].columns:
                        df_ema[c] = np.expand_dims(np.asarray( df_ema[ c ] ).astype(float),1) * np.asarray( df_ema[ vol_scales[strat][cntr].columns ] ).astype(float)

                # df_ema now has the weights (EMA * vola scaling). vola scaling col is passive BM

                # i) take lags of weights & ii) merge instrument returns to it
                df_ema = df_ema.shift( periods = signal_lag )
                # rename remaining cols
                df_ema.rename(columns = colmapper, inplace = True)
                if strat=='FI':
                    df_ema['NN'] = df_ema['NN'] * -1 # FI is reversal (yields as supervision)
                df_instr_rets = instr_rets[strat][cntr]
                df_instr_rets = df_instr_rets.loc[df_instr_rets.index >= cutoff].fillna(0.0)
                df_instr_rets.rename(columns = {x: 'return' for x in df_instr_rets.columns} , inplace = True)
                df_instr_rets = df_instr_rets.merge(df_ema, on = ['DATE'], how = 'left').fillna(method='ffill')
                # ii) organize to (Date, instr, wgt, return) col structure ( loop over countries )
                df_instr_rets['instrument'] = cntr + '_' + strat

                bt_dfs.append(df_instr_rets)


        bt_df = pd.concat(bt_dfs)
        bt_df.groupby(by='instrument').size()
        # organize into BT format
        bt_df.index.name = 'Date'


        #c = 'NN'
        bts = {}
        for c in colmapper.values():
            df = deepcopy( bt_df[['return',c,'instrument']] )
            df.rename(columns = {c:'weight'},inplace = True)
            df['weight'] = df['weight'].fillna(method = 'ffill').fillna(0.0) # forward fill + zero pad
            bti = bt(df,Name=c)
            bti.fill_rebal_weights(rebal_days_week= [0,1,2,3,4] )
            bti.calc_returns()
            bti.change_expostvola(0.01)

            bts[c] = bti



        bts_all[fastsignal] = bts


    if save_res:
        # pickle dump the dicts
        # -> has been expanding window demeaned already!
        with open('data/workspace/backtests/'+NN_prefix+'BT'+('_').join(strat_included) +'_l' + str(signal_lag) + '_'+reshuffling+'.pickle', "wb") as output_file:
            pckl.dump(bts_all, output_file)

    bts_strat[''.join(strat_included)] = bts_all



if save_res_all:
    # save all strats
    with open('data/workspace/backtests/'+NN_prefix+'BTallstrats'+'_l'+str(signal_lag)+'_'+reshuffling+'.pickle', "wb") as output_file:
        pckl.dump(bts_strat, output_file)





#######################################################################################################################
#######################################################################################################################
# analysis part
#######################################################################################################################
#######################################################################################################################
# set GRU prefix
GRU_prefix = 'GRUAE_d_'
# load back
#with open('data/workspace/backtests/BT'+('_').join(strat_included)+  '_EMAf' +str(EMA['fast']) + '_EMAs' +
#          str(EMA['slow']) + '_l' + str(signal_lag) + '.pickle', "rb") as output_file:
#    bts = pckl.load( output_file )
NN_prefixes = [ 'ReconFirst_','PureClassif_d_','PLSreg_d_',GRU_prefix,'PrCA_d_']
files = ['data/workspace/backtests/'+NNprefix+'BTallstrats'+'_l'+str(signal_lag)+'_'+reshuffling+'.pickle' for NNprefix in NN_prefixes]  # have the biweekly!
NN_prefixes = [ '','PureClassif_d_','PLSreg_d_',GRU_prefix,'PrCA_d_']
global_colnames = [ 'Fs',   'idx', 'Passive'] + ['NN'+n for n in NN_prefixes]
bts_strat = {}

for i, f in enumerate(files):
    with open(f, "rb") as output_file:
        bts_strat_i = pckl.load( output_file)
    for k in bts_strat_i.keys():
        if i==0:
            bts_strat[k] = bts_strat_i[k]
        else:
            for kk in bts_strat_i[k].keys():
                newname = 'NN'+NN_prefixes[i]
                bts_strat[k][kk][newname] = bts_strat_i[k][kk]['NN']





# t to t regression...
bts_all = bts_strat['EQFIFX']

# collect market data
# the market return EQ: row average of
eqmkt = eqfutrets.mean( axis = 1 )
# FX
fxmkt = sentdfs['US'][sentdata_dict['US']['FX']]['FX_DXYcrncyRet']
# FI: row average of
fimkt = firets.mean( axis = 1)
# commodity market (optional)
cmdty = sentdfs['US'][sentdata_dict['US']['cmdty']]['c_CRBcmdtyRet']

pd_mkt = pd.merge(eqmkt.rename('eqmkt'), fxmkt.rename('fxmkt'), left_index=True,right_index=True,how = 'outer')
pd_mkt = pd.merge(pd_mkt, fimkt.rename('fimkt'), left_index=True,right_index=True,how = 'outer')
pd_mkt = pd.merge(pd_mkt, cmdty.rename('cmdty'), left_index=True,right_index=True,how = 'outer')
pd_mkt = pd_mkt.fillna(value = 0.0)

# collect tables
# alpha tstats
df_tstats = pd.DataFrame(columns=list(itertools.product(list(bts_strat.keys()), global_colnames)), index=list(bts_all.keys()))
# IR
df_IR = pd.DataFrame(columns=list(itertools.product(list(bts_strat.keys()), global_colnames)), index=list(bts_all.keys()))


# the following: only for EQFXFI:---
df_IR_contract = pd.DataFrame(columns=['IR'],
                              index=list(bts_all[list(bts_all.keys())[0]][global_colnames[0]].BTdata.instrument.unique()),data = 0.0)
BT_sh = pd.DataFrame(columns=global_colnames,
                              data=np.zeros((len(bts_all[list(bts_all.keys())[0]][global_colnames[0]].BTdata_ret.index ),len(global_colnames))),
                              index = bts_all[list(bts_all.keys())[0]][global_colnames[0]].BTdata_ret.index )
BT_lg = pd.DataFrame(columns=global_colnames,
                              data=np.zeros((len(bts_all[list(bts_all.keys())[0]][global_colnames[0]].BTdata_ret.index),len(global_colnames))),
                              index = bts_all[list(bts_all.keys())[0]][global_colnames[0]].BTdata_ret.index )


# long window short window collection
short_term_IR_threshold = 10
strat_of_interest_short = 'NN'
long_term_IR_threshold = 4*21

for b in bts_strat.keys():
    # set BT to be used
    bts_all = bts_strat[b]

    for c in global_colnames:
        # collect t-stats
        for k in bts_all.keys():

            # backtest returns
            Y_bt = bts_all[k][c].BTdata_ret['ret']

            # join to backtest returns
            pd_mkt_YX = pd.merge(Y_bt.rename('BT'), pd_mkt, how = 'left', left_index=True, right_index=True)
            # data.set_index(['EntityNo', 'Time'], inplace=True)

            # collect t-stats of alphas
            #exog_vars = ["black", "hisp", "exper", "expersq", "married", "educ", "union", "year"]
            #exog = sm.add_constant(data[exog_vars])
            #mod = PooledOLS(data.lwage, exog)
            #pooled_res = mod.fit()
            #print(pooled_res)
            exog = sm.add_constant( pd_mkt_YX[['eqmkt','fxmkt','fimkt']] )
            OLS = sm.OLS(pd_mkt_YX['BT'], exog).fit()
            # OLS.summary()

            # collect t-stats
            df_tstats[(b,c)][k] = OLS.tvalues['const']

            # collect IRs
            _, _, IR = bts_all[k][c].get_IR(bts_all[k][c].BTdata_ret, 252, 'ret', 'EQline ' + bts_all[k][c].Name)
            df_IR[(b,c)][k] = IR

            # collect short window / long window backtests (only for EQFXFI)
            if b=='EQFIFX':
                if k<= short_term_IR_threshold and c == strat_of_interest_short:
                    bts_all[k][c].set_instr_returnsTable()
                    df_contract = bts_all[k][c].get_stats_per_instr()
                    for ix in df_contract.index:
                        df_IR_contract['IR'][ix] += df_contract['IR'][ix] / sum([x<= short_term_IR_threshold for x in bts_all.keys()])
                if k <= short_term_IR_threshold:
                    # collect and average the backtest strats (short and long version)
                    BT_sh[c] += bts_all[k][c].BTdata_ret['ret'].values / sum([x<= short_term_IR_threshold for x in bts_all.keys()])
                if k >= long_term_IR_threshold:
                    BT_lg[c] += bts_all[k][c].BTdata_ret['ret'].values/ sum([x>= long_term_IR_threshold for x in bts_all.keys()])




for c in BT_lg.columns:
    BT_lg[c] = BT_lg[c]/ (BT_lg[c].std()*np.sqrt(252))*0.01
    BT_lg[c+' EQline'] = BT_lg[c].cumsum()+1.0
    BT_sh[c] = BT_sh[c] / (BT_sh[c].std() * np.sqrt(252)) * 0.01
    BT_sh[c+' EQline'] = BT_sh[c].cumsum() + 1.0
# rename
# BT_sh['NN'+' EQline'].plot(); BT_lg['Fs'+' EQline'].plot(); BT_lg['Passive'+' EQline'].plot();
NNname_map = dict(zip(global_colnames,['Return Trend','idx','Passive','supervised AE trend','NN Class. trend','PLS trend','GRU trend','PCA trend' ]))
BT_sh.drop('Passive',axis = 1,inplace =True)
BT_sh.rename(columns = {x+' EQline': NNname_map[x] for x in global_colnames}, inplace = True)
BT_lg.drop('Passive',axis = 1,inplace =True)
BT_lg.rename(columns = {x+' EQline': NNname_map[x] for x in global_colnames}, inplace = True)


contract_map = {'AU_EQ':'SPI200' ,'CA_EQ':'S&P/TSE60Canada','CN_EQ':'HangSeng','EU_EQ': 'EUROSTOXX50','GB_EQ':'FTSE100','JP_EQ':'TOPIX',
                'US_EQ':'S&P500Mini',
                'AU_FI':'AU10Yr','CA_FI':'Can10Yr','EU_FI':'Bund', 'GB_FI':'Long Gilt', 'JP_FI': 'JP10Yr', 'US_FI':'US10Yr',
                'AU_FX': 'AUD/USD','CA_FX':'CAD/USD','EU_FX': 'EUR/USD','GB_FX':'GBP/USD','JP_FX':'JPY/USD'  }
# plot IR per contract, backtest
df_contract['instruments'] = ''
for x in contract_map.keys():
    df_contract['instruments'][x] = contract_map[x]
df_contract = df_contract.loc[list(contract_map.keys())] # reorder
df_contract.set_index('instruments',inplace = True)






# short and long IR table
bt_c = bt(0.0)
ls = pd.DataFrame(data = np.zeros((2,6)), columns = ['PCA','PLS','Sup.AE','NN Class.','GRU','Return'],index = ['short term','long term'])
ls.loc['short term']['NN Class.'] = bt_c.get_IR(BT_sh, 252, 'NNPureClassif_d_', NNname_map['NNPureClassif_d_'])[2]
ls.loc['short term']['Sup.AE'] = bt_c.get_IR(BT_sh, 252, 'NN', 'supervised AE trend')[2]
ls.loc['short term']['PLS'] = bt_c.get_IR(BT_sh, 252, 'NNPLSreg_d_', NNname_map['NNPLSreg_d_'])[2]
ls.loc['short term']['GRU'] = bt_c.get_IR(BT_sh, 252, 'NN'+GRU_prefix, NNname_map['NN'+GRU_prefix])[2]
ls.loc['short term']['PCA'] = bt_c.get_IR(BT_sh, 252, 'NNPrCA_d_', NNname_map['NNPrCA_d_'])[2]
ls.loc['short term']['Return'] = bt_c.get_IR(BT_sh, 252, 'Fs', 'Return Trend')[2]

ls.loc['long term']['Sup.AE'] = bt_c.get_IR(BT_lg, 252, 'NN', 'supervised AE trend')[2]
ls.loc['long term']['NN Class.'] = bt_c.get_IR(BT_lg, 252, 'NNPureClassif_d_', NNname_map['NNPureClassif_d_'])[2]
ls.loc['long term']['PLS'] = bt_c.get_IR(BT_lg, 252, 'NNPLSreg_d_', NNname_map['NNPLSreg_d_'])[2]
ls.loc['long term']['GRU'] = bt_c.get_IR(BT_lg, 252, 'NN'+GRU_prefix, NNname_map['NN'+GRU_prefix])[2]
ls.loc['long term']['PCA'] = bt_c.get_IR(BT_lg, 252, 'NNPrCA_d_', NNname_map['NNPrCA_d_'])[2]
ls.loc['long term']['Return'] = bt_c.get_IR(BT_lg, 252, 'Fs', 'Return Trend')[2]
ls = ls.round(2)
ls.to_latex(os.path.expanduser('~/Downloads/DeepArticle/tables/') + 'BT_IR_ls_2023.tex', escape = False, index = True)

# reorganize df_tstats and save as tex file
df_tstats_out = pd.DataFrame(columns=[('ret. trend','EQ\&FI\&FX'),
                                      ('ret. trend','EQ'),('ret. trend','FI'),('ret. trend','FX'),
                                      ('sup. AE trend','EQ\&FI\&FX'),
                                      ('sup. AE trend','EQ'),('sup. AE trend','FI'),('sup. AE trend','FX')]
                             , index=list(bts_all.keys()))
df_tstats_out[('sup. AE trend','EQ\&FI\&FX')] = df_tstats[('EQFIFX','NN')]  #
df_tstats_out[('ret. trend','EQ\&FI\&FX')] = df_tstats[('EQFIFX','Fs')]  #
df_tstats_out[('sup. AE trend','EQ')] = df_tstats[('EQ','NN')]  # use EQ backtests
df_tstats_out[('sup. AE trend','FI')] = df_tstats[('FI','NN')]  # use FX backtests
df_tstats_out[('sup. AE trend','FX')] = df_tstats[('FX','NN')]  # use FI backtests
df_tstats_out[('ret. trend','EQ')] = df_tstats[('EQ','Fs')]  # use EQ backtests
df_tstats_out[('ret. trend','FI')] = df_tstats[('FI','Fs')]  # use FX backtests
df_tstats_out[('ret. trend','FX')] = df_tstats[('FX','Fs')]  # use FI backtests
df_tstats_out.columns = pd.MultiIndex.from_tuples(df_tstats_out.columns)

df_tstats_out = df_tstats_out.astype(np.float32)
df_tstats_out = df_tstats_out.round(2)
df_tstats_out.to_latex(os.path.expanduser('~/Downloads/DeepArticle/tables/') + 'BT_tstats_2023.tex', escape = False, index = True)

# add IR to the tstats, textbf for significant t-stats
df_IR_out = pd.DataFrame(columns=[('ret. trend','EQ\&FI\&FX'),('sup. AE trend','EQ\&FI\&FX'),
                                  ('ret. trend','EQ'),('ret. trend','FI'),('ret. trend','FX'),
                                      ('sup. AE trend','EQ'),('sup. AE trend','FI'),('sup. AE trend','FX')], index=list(bts_all.keys()))
df_IR_out[('sup. AE trend','EQ\&FI\&FX')] = df_IR[('EQFIFX','NN')]  #
df_IR_out[('ret. trend','EQ\&FI\&FX')] = df_IR[('EQFIFX','Fs')]  #
df_IR_out[('sup. AE trend','EQ')] = df_IR[('EQ','NN')]  # use EQ backtests
df_IR_out[('sup. AE trend','FI')] = df_IR[('FI','NN')]  # use FX backtests
df_IR_out[('sup. AE trend','FX')] = df_IR[('FX','NN')]  # use FI backtests
df_IR_out[('ret. trend','EQ')] = df_IR[('EQ','Fs')]  # use EQ backtests
df_IR_out[('ret. trend','FI')] = df_IR[('FI','Fs')]  # use FX backtests
df_IR_out[('ret. trend','FX')] = df_IR[('FX','Fs')]  # use FI backtests

df_IR_out.columns = pd.MultiIndex.from_tuples(df_IR_out.columns)

df_IR_out = df_IR_out.astype(np.float32)
df_IR_out = df_IR_out.round(2)
df_IR_out.to_latex(os.path.expanduser('~/Downloads/DeepArticle/tables/') + 'BT_IR_2023.tex', escape = False, index = True)


# combine IR and tstats into one table,  as tuples (?)
df_str = df_IR_out.astype(str)
for x in df_tstats_out.columns:
    #
    df_str[x].loc[df_IR_out[x] == df_IR_out[x].max()] = '$\textbf{'+df_str[x].loc[df_IR_out[x] == df_IR_out[x].max()] +'}$'

    ind_10perc = np.logical_and(np.abs(df_tstats_out[x]) > 1.65, np.abs(df_tstats_out[x]) <= 1.96)
    df_str[x].loc[ind_10perc] = df_str[x].loc[ind_10perc].apply(lambda x: x + '$^{\\ast}$')
    ind_5perc = np.logical_and(np.abs(df_tstats_out[x]) > 1.96, np.abs(df_tstats_out[x]) <= 2.58)
    df_str[x].loc[ind_5perc] = df_str[x].loc[ind_5perc].apply( lambda x: x + '$^{\\ast\\ast}$')

    #tex_out[y][model_name][ind_5perc] = tex_out[y][model_name][ind_5perc].apply(lambda x: x + '$^{\\ast\\ast}$')
    #tex_out[y][model_name][np.abs(tstats) > 2.58] = tex_out[y][model_name][np.abs(tstats) > 2.58].apply(



    df_str[x].loc[np.abs(df_tstats_out[x]) > 2.58] = df_str[x].loc[np.abs(df_tstats_out[x]) > 2.58].apply(lambda x: x + '$^{\\ast\\ast\\ast}$')

df_str.columns = pd.MultiIndex.from_tuples([(   'return', 'eq.-wgt.'),
            ('sup. AE', 'eq.-wgt.'),
            (   'return',         'EQ'),
            (   'return',         'FI'),
            (   'return',         'FX'),
            ('sup. AE',         'EQ'),
            ('sup. AE',         'FI'),
            ('sup. AE',         'FX')])
df_str[[(   'return', 'eq.-wgt.'),
            (   'return',         'EQ'),
            (   'return',         'FI'),
            (   'return',         'FX'),
        ('sup. AE', 'eq.-wgt.'),
            ('sup. AE',         'EQ'),
            ('sup. AE',         'FI'),
            ('sup. AE',         'FX')]].to_latex(os.path.expanduser('~/Downloads/DeepArticle/tables/') + 'BT_tstatsIR_2023.tex', escape = False, index = True)

