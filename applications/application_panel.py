import pickle as pckl
import pandas as pd
import applications.BTdata_utils as BTdata_utils
import datetime
import numpy as np
from linearmodels import PanelOLS
# Durbin-Watson-Test
from statsmodels.stats.stattools import durbin_watson
from linearmodels import RandomEffects
from linearmodels import PooledOLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from collections import defaultdict
import os

# if __name__=='__main__':
# add GDP growth always
# quarterly freq, GDP growth rates (ex China)  ->  dependent var of midas exercise
with open('data/workspace/midasY.pickle', "rb") as output_file:
    midasY = pckl.load( output_file)

############################################################################################################
# params
############################################################################################################
add_binary = True
# general cutoff
cutoff_low = datetime.datetime.strptime( '2004-01-01', '%Y-%m-%d') #'2004-01-01' '2017-12-31'  endoftraining = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')  #"2021-12-24"
cutoff_high = datetime.datetime.strptime('2022-01-01', '%Y-%m-%d')
dim_of_supervision = 5
# can have daily / weekly panel and monthly panel!
frq = 'd'
# 1 - collect data on monthly frequency
if frq=='m':
    pos, pos_cmdt, pos_vol,pos_i = 2,3,2,2  # positions of monthly column names in lists in sentdata_dict
    #sent_slot_pos = { 'var0': 0, 'var1':3}
    frqsuffix = '_mth'
elif frq == 'w':
    pos, pos_cmdt, pos_vol,pos_i = 1, 2, 1,1  # positions of monthly column names in lists in sentdata_dict
    # sent_slot_pos = { 'var0': 0, 'var1':3}
    frqsuffix = '_wk'
else: # 2 - collect data on daily freq
    pos, pos_cmdt,pos_vol, pos_i = 0,0,2,1
    frqsuffix = ''

lags_in = 3 # variables included
lags_discard = 2 # for tex output
# slot positions
sent_slot_pos = {'var'+str(x)+ '_{'+frq : x for x in range(0,dim_of_supervision)}

# run pooled ols + tests
run_pooled_tests = False

# collect all prefixes
NN_prefixes = { 'SENT_PC':'sent-PCA', 'M_PC':'macro-PCA', 'ReconFirst_':'SupAE',
                'PureClassif_d_':'NNclssfr',
                'GRUAE_d_':'GRU',
                'PureRegr_d_':'NNregr',
                'PrCA_d_':'PCAroll','PLS01_d_':'PLSbin','PLSreg_d_':'PLSregr'
                }

# loop over files and descriptions each file contains learned representations ...
file_des_tpls = [
    ('data/workspace/sent_NNresults_test_semiannual_retrain_lead0o_weightdecay0_0001.pickle','d'),
    ('data/workspace/sent_NNresults_test_semiannual_retrain_lead0m.pickle','k'),  # j/k -> cb  (GRU, GRUAE)
    ('data/workspace/sent_NNresults_test_semiannual_retrain_lead0e.pickle','e'),
    ('data/workspace/sent_NNresults_test_semiannual_retrain_lead0f.pickle','f'),
    ('data/workspace/sent_NNresults_test_semiannual_retrain_lead0logregL1.pickle','r')
]
# selection of sentiment slot (NN = neuralnet)
sent_slot = 'NN'
############################################################################################################




'''
loop over files 
'''
EQprefix = '\text{EQ: }r_{'
FIprefix = '\text{FI: } \Delta i_{'
FXprefix = '\text{FX: }r_{'
EQFIFX_prefix_dict = {EQprefix: 'EQ',FIprefix: 'FI', FXprefix: 'FX'}

# inititialize tex files
tex_out = defaultdict()
for i,(f,desc) in enumerate(file_des_tpls):
    print('iteration: '+str(i) +', file,descr.: '+ f + ',' + desc )
    # load file
    with open(f, "rb") as output_file:
        sentdfs = pckl.load(output_file)

    if i==0:
        NN_prefixes_loop = list(NN_prefixes.keys())
    else:
        NN_prefixes_loop = [x for x in list(NN_prefixes.keys()) if x not in ['SENT_PC', 'M_PC']] # cut off Macro PCA and sent PCA (multiple occurrences)

    # add gdp growth
    BTdata_utils.merge_DATAtoRP(sentdfs, midasY, fill_method='ffill')

    for j,NN_prefix in enumerate(NN_prefixes_loop):
        print('prefix: '+NN_prefix)
        #NN_prefix = NN_prefixes[0]
        # loop over prefixes to get cols
        sentdata_dict = BTdata_utils.get_colname_dicts(sentdfs, dataset_str_patterns={'eqidx': 'EQidx_(.*?)','eqidxvol': 'EQidxvol_(.*?)',
                                                                                      'fx': 'FX_(.*?)','cmdty': 'c_(.*?)','gdp':  'orig_(.*?)', # 'M_PC0_mth_30(.*?)',
                                                                                      # already in perc change QoQ
                                                                                      'gdp_cn': 'M_dseas_Index of industrial(.*?)',
                                                                                      'macro': 'M_(.*?)',
                                                                                      'sentPCA': 'SENT_(.*?)','NN': NN_prefix + '(.*?)',
                                                                                      'ir': 'ZYLD_(.*?)',
                                                                                      # 'IR3M_(.*?)',
                                                                                      # inflation indicators (already in MoM percentage changes)
                                                                                      'P_US': 'M_dseas_Consumer (.*?)','P_EU': 'M_dseas_HICP_(.*?)',
                                                                                      'P_GB': 'M_dseas_Consumer (.*?)','P_JP': 'M_dseas_Consumer (.*?)',
                                                                                      'P_CA': 'M_dseas_Consumer (.*?)','P_AU': 'M_dseas_Consumer (.*?)',
                                                                                      'P_CN': 'M_dseas_Consumer (.*?)',
                                                                                      })
        # construct data set by country and merge it to panel data set
        if len(sentdata_dict['US'][sent_slot]) > 0:  # US required
            for ii, n in enumerate(sentdfs.keys()):

                if n in ['US','EU','JP','CA','GB','AU']:
                    core_cols = [sentdata_dict[n]['eqidx'][pos],sentdata_dict[n]['eqidx'][-2],sentdata_dict[n]['eqidxvol'][pos_vol],
                                           sentdata_dict[n]['fx'][pos],sentdata_dict[n]['cmdty'][pos_cmdt],
                                 sentdata_dict[n]['gdp'][0], # questionable
                                           sentdata_dict[n]['ir'][pos_i],
                                 sentdata_dict[n]['P_'+n][0] # questionable
                                 ]
                    core_cols_new_names = [EQprefix+frq,EQprefix+'y','\text{EQ: vol}_{m',FXprefix+frq,'\text{CMDT: }r_{'+frq,
                                           '\Delta GDP_{m',
                                           FIprefix+frq,
                                           'P_{m'
                                           ]
                    sentcols = [sentdata_dict[n][sent_slot][x] + frqsuffix for x in sent_slot_pos.values()]
                    panel_i = sentdfs[n][core_cols + sentcols]
                    sentcols_new_names = [x for x in sent_slot_pos.keys()]
                    panel_i.columns = core_cols_new_names + sentcols_new_names



                if n == 'CN':
                    core_cols = [sentdata_dict[n]['eqidx'][pos],sentdata_dict[n]['eqidx'][-2],sentdata_dict[n]['eqidxvol'][2],
                                           sentdata_dict[n]['cmdty'][pos_cmdt],sentdata_dict[n]['gdp_cn'][0],
                                           sentdata_dict[n]['P_'+n][0] ]
                    sentcols = [sentdata_dict[n][sent_slot][x] +frqsuffix for x in sent_slot_pos.values() ]
                    panel_i = sentdfs[n][ core_cols + sentcols ]
                    sentcols_new_names = [x for x in sent_slot_pos.keys()]
                    panel_i.columns = [EQprefix+frq, EQprefix+'y', '\text{EQ: vol}_{m', '\text{CMDT: }r_{'+frq,
                                       '\Delta GDP_{m',
                                       'P_{m'
                                       ] + sentcols_new_names
                    # add missing cols
                    panel_i[FIprefix+frq] = 0.0; panel_i[FXprefix+frq] = 0.0

                # rescale
                panel_i[sentcols_new_names] = panel_i[sentcols_new_names] / 10
                panel_i['region'] = n # add i

                if frq == 'm': # get last day of the month
                    panel_i['day'] = panel_i.index.day # add day
                    panel_i = panel_i.loc[ panel_i.groupby([(panel_i.index.year), (panel_i.index.month)])['day'].transform(max) == panel_i['day']]
                    panel_i.drop(['day'],axis = 1, inplace = True)
                if frq =='w': # get last day of week
                    panel_i['day'] = panel_i.index.day  # add day
                    panel_i = panel_i.loc[
                        panel_i.groupby([(panel_i.index.year), (panel_i.index.month), (panel_i.index.week)])['day'].transform(max) == panel_i['day']]
                    panel_i.drop(['day'], axis=1, inplace=True)

                # take lags of variables
                Xvars = [EQprefix+frq,EQprefix+'y','\text{EQ: vol}_{m',FXprefix+frq,'\text{CMDT: }r_{'+frq,
                         '\Delta GDP_{m',FIprefix+frq,'P_{m'] + sentcols_new_names
                for x in Xvars:
                    for k in range(0,lags_in):
                        panel_i[x+',t-'+str(k+1) +'}'] = panel_i[x].shift(k+1,axis = 0,fill_value = 0.0) # first lag

                panel_i = panel_i.loc[np.logical_and(panel_i.index > cutoff_low,panel_i.index < cutoff_high) ]  # cutoff

                # set index to Date, region
                panel_i.set_index('region', append = True, inplace= True)
                if ii == 0:
                    pnl = panel_i
                else:
                    pnl = pnl.append(panel_i)

            # MultiIndex DataFrames where the outer index is the entity and the inner is the time index.
            pnl = pnl.swaplevel(0,1)

            # fill nas
            pnl = pnl.fillna(0.0)

            Xvars = [EQprefix+ frq, EQprefix+'y', '\text{EQ: vol}_{m', FXprefix + frq, '\text{CMDT: }r_{' + frq,
                     #'\Delta GDP_{m',
                     FIprefix + frq,
                     #'P_{m'
                     ] + sentcols_new_names

            Xs_discard = []
            for y in [EQprefix, FIprefix, FXprefix]:
                # loop over EQ, FI, FX:
                # construct and add binary vars
                for k in range(0, lags_in):
                    pnl[y + frq + ',t-'+str(k+1) +'}' + 'bnry'] = (pnl[y + frq + ',t-'+str(k+1) +'}' ] >0 )*1
                Xs= []
                for k in range(0, lags_in):
                    if add_binary:
                        Xs += [x+',t-'+str(k+1)+'}' for x in Xvars ] + [ y + frq + ',t-'+str(k+1)+'}bnry']
                        Xs_discard += [x+',t-'+str(k+1)+'}' for x in Xvars if k>=lags_discard ]
                        if k>=lags_discard: Xs_discard += [ y + frq + ',t-'+str(k+1)+'}bnry'   ]
                    else:
                        Xs += [x+',t-'+str(k+1)+'}' for x in Xvars ]
                        Xs_discard += [x + ',t-' + str(k + 1) + '}' for x in Xvars if k >= lags_discard]


                # exclude sent vars for benchmarking
                #Xs = [x+'_lg1' for x in Xvars if x not in list(sent_slot_pos.keys())  ] +  [x+'_lg2' for x in Xvars if x not in list(sent_slot_pos.keys()) ]

                if i==0 and j==0: # save only once
                    # save pnl table
                    pnl.to_csv( 'data/workspace/'+EQFIFX_prefix_dict[y] + 'paneldata.csv' )

                if run_pooled_tests:
                    # 3.B Non-Autocorrelation / homosced. in pooledOLS
                    exog = sm.add_constant(pnl[Xs])
                    pooled_OLS = PooledOLS(pnl[y+frq], exog).fit(cov_type='clustered', cluster_entity=True)
                    # durbin watson (non-autocorr necessary condition for pooled ols)
                    durbin_watson_test_results = durbin_watson(pooled_OLS.resids)
                    print(durbin_watson_test_results)
                    # The closer to 0 the statistic, the more evidence for positive serial correlation.
                    # The closer to 4, the more evidence for negative serial correlation.

                    # white test of homosced
                    pooled_OLS_dataset = pd.concat([pnl[Xs], pooled_OLS.resids], axis=1)
                    exog = sm.tools.tools.add_constant(pnl[Xs]).fillna(0)
                    white_test_results = het_white(pooled_OLS_dataset['residual'], exog)
                    labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val']
                    # ->
                    print(dict(zip(labels, white_test_results)))  # p < 0.05 -> heterosced indicated
                # hausman test of endogeneity
                re = RandomEffects(pnl[y+frq], pnl[Xs]).fit(cov_type="kernel")   # cov_type   ...  "kernel" -> Newey west .. "robust" -> white
                fe = PanelOLS(pnl[y+frq], pnl[Xs], entity_effects=True, check_rank=False).fit(cov_type="kernel")  #
                chi2, df, pval = BTdata_utils.hausman(fe,re)
                if pval < 0.05:
                    print('fixed effects according to hausman test')
                    selected = fe
                else:
                    # we are almost always here:
                    print('random effects according to hausman test')
                    selected = re
                # print(selected.summary)
                # name of "run"
                model_name = desc + ':' + NN_prefixes[NN_prefix]
                # construct tex (i==0, j==0) -> tex table empty
                tex_table_names = ['$' + x + '$' for x in selected._var_names]
                # ljung box lag 10
                lb = sm.stats.acorr_ljungbox(selected.resids, lags=[2], return_df=True)
                tex_out_stats = pd.DataFrame({model_name: [(round(selected.rsquared*100,1), round(selected.rsquared_between*100,1),
                                                            round(selected.rsquared_within*100,1)),
                                                           (round(selected.loglik),round(float(lb.lb_pvalue.values),2)) ]},
                    index=['$R^2$,$R^2_{btw}$,$R^2_{wthn}$','loglik, $LB_{10}(p)$',])
                if i==0 and j==0:
                    # -> organize this in pandas, convert cols to string, use t-stat info to attach asterisks
                    tex_out[y] = pd.DataFrame({model_name: [round(x*10,2) for x in selected.params]}, index=tex_table_names)
                    # append R2s
                    tex_out[y] = tex_out[y].append(tex_out_stats)
                else:
                    tex_out_i = pd.DataFrame({model_name: [round(x*10,2) for x in selected.params]}, index=tex_table_names)
                    # append R2s
                    tex_out_i = tex_out_i.append(tex_out_stats)
                    tex_out[y] = tex_out[y].merge(tex_out_i, right_index=True,left_index=True)
                # modify tstats
                tstats = [x for x in selected.tstats] + [0.0]*tex_out_stats.shape[0]
                # to string
                tex_out[y][model_name] = tex_out[y][model_name].astype(str)
                ind_10perc = np.logical_and(np.abs(tstats) > 1.65, np.abs(tstats) <= 1.96)
                tex_out[y][model_name][ind_10perc] = tex_out[y][model_name][ind_10perc].apply(lambda x: x + '$^{\\ast}$')
                ind_5perc = np.logical_and(np.abs(tstats) > 1.96, np.abs(tstats) <= 2.58)
                tex_out[y][model_name][ind_5perc] = tex_out[y][model_name][ind_5perc].apply(lambda x: x + '$^{\\ast\\ast}$')
                tex_out[y][model_name][np.abs(tstats) > 2.58] = tex_out[y][model_name][np.abs(tstats) > 2.58].apply(
                    lambda x: x + '$^{\\ast\\ast\\ast}$')



# organize tex and results

# only 2 lags max reported
rowname_map = {
    '$'+FXprefix+'d,t-1}bnry$'  : '$\text{FX: }$.$\text{sgn}(r_{d,t-1})$',
    '$'+FXprefix+'d,t-2}bnry$'  : '$\text{FX: }$.$\text{sgn}(r_{d,t-2})$',
    '$'+FIprefix+'d,t-1}bnry$'  : '$\text{FI: }$.$\text{sgn}(r_{d,t-1})$',
    '$'+FIprefix+'d,t-2}bnry$'  : '$\text{FI: }$.$\text{sgn}(r_{d,t-2})$',
    '$'+EQprefix+'d,t-1}bnry$'  : '$\text{Eq: }$.$\text{sgn}(r_{d,t-1})$',
    '$'+EQprefix+'d,t-2}bnry$'  : '$\text{Eq: }$.$\text{sgn}(r_{d,t-2})$',
    # std
    '$'+FXprefix+'d,t-1}$'  : '$\text{FX: }$.$r_{d,t-1}$',
    '$'+FXprefix+'d,t-2}$'  : '$\text{FX: }$.$r_{d,t-2}$',
    '$'+FIprefix+'d,t-1}$'  : '$\text{FI: }$.$r_{d,t-1}$',
    '$'+FIprefix+'d,t-2}$'  : '$\text{FI: }$.$r_{d,t-2}$',
    '$'+EQprefix+'d,t-1}$'  : '$\text{Eq: }$.$r_{d,t-1}$',
    '$'+EQprefix+'d,t-2}$'  : '$\text{Eq: }$.$r_{d,t-2}$',
    '$\text{CMDT: }r_{d,t-1}$'  : '$\text{Cmdt: }$.$r_{d,t-1}$',
    '$\text{CMDT: }r_{d,t-2}$'  : '$\text{Cmdt: }$.$r_{d,t-2}$',
    '$\text{EQ: vol}_{m,t-1}$': '$\text{Eq: }$.$\text{vol}_{m,t-1}$',
    '$\text{EQ: vol}_{m,t-2}$': '$\text{Eq: }$.$\text{vol}_{m,t-2}$',
    '$\text{EQ: }r_{y,t-1}$': '$\text{Eq: }$.$r_{y,t-1}$',
    '$\text{EQ: }r_{y,t-2}$': '$\text{Eq: }$.$r_{y,t-2}$',
    # sentiments -> S
    '$var0_{d,t-1}$' :  '$\text{R-Eq/PC1: }$.$S_{d,t-1}$'  , # EQ
    '$var1_{d,t-1}$' :   '$\text{R-FI/PC2: }$.$S_{d,t-1}$'  , # FI
    '$var2_{d,t-1}$' : '$\text{R-Macro/PC3: }$.$S_{d,t-1}$'  , #macr
    '$var3_{d,t-1}$' : '$\text{R-Cmdt/PC4: }$.$S_{d,t-1}$'  , # cmdty
    '$var4_{d,t-1}$' : '$\text{R-FX/PC5: }$.$S_{d,t-1}$'  , # FX
    '$var0_{d,t-2}$' :  '$\text{R-Eq/PC1: }$.$S_{d,t-2}$'  , # EQ
    '$var1_{d,t-2}$' :   '$\text{R-FI/PC2: }$.$S_{d,t-2}$'  , # FI
    '$var2_{d,t-2}$' : '$\text{R-Macro/PC3: }$.$S_{d,t-2}$'  , #macr
    '$var3_{d,t-2}$' : '$\text{R-Cmdt/PC4: }$.$S_{d,t-2}$'  , # cmdty
    '$var4_{d,t-2}$' : '$\text{R-FX/PC5: }$.$S_{d,t-2}$'  , # FX
    '$R^2$,$R^2_{btw}$,$R^2_{wthn}$':'Stats:.$R^2$,$R^2_{btw}$,$R^2_{wthn}$' ,
    'loglik, $LB_{10}(p)$': 'Stats:.loglik, $LB_{10}(p)$'
}
Xs_discard = list(set(['$'+x+'$' for x in Xs_discard ]))


# redo the colname map
# apply
final_colname_map = dict(zip(['d:sent-PCA', 'd:macro-PCA', 'd:SupAE', 'd:NNclssfr', 'k:GRU',
        'g:NNclssfrshllw'
       'i:NNclssfrdeeper', 'p:PCAroll', 'p:PLSbin', 'p:PLSregr'   ],
                        ['d:sent-PCA', 'd:macro-PCA', 'Sup.AE', 'NNclass', 'GRU',
       'g:NNclssfrshllw',
       'i:NNclssfrdeeper', 'PCA', 'PLSbin', 'PLS'  ]  ) )

# selection:
colselection = [  'PCA', 'PLS', 'Sup.AE', 'NNclass', 'GRU' ]  # 'PLSbin',

# blocks of x:
blocksize = 5
discard_lags = 2 # discard lags >= discard_lags
# tex table out to tex file
for y in [EQprefix, FIprefix, FXprefix]:

    # change cols:
    texoutcols = list(tex_out[y].columns)
    for c in texoutcols:
        tex_out[y].rename( columns = {c:final_colname_map[c]}, inplace = True)
    # selection of certain cols
    tex_out[y] = tex_out[y][colselection]

    # discard lags > x
    splits = [ i for i in range(0,tex_out[y].shape[1],blocksize)] + [tex_out[y].shape[1]]


    for k in range(0,(len(splits)-1)):
        col_sel = tex_out[y].columns[splits[k]:splits[k+1]]
        fn = y[5:7]
        tb = tex_out[y][col_sel]
        tb = tb.loc[[x for x in tb.index if x not in Xs_discard]]
        tb.index.name = fn

        test_stats = tb.loc[[ '$R^2$,$R^2_{btw}$,$R^2_{wthn}$','loglik, $LB_{10}(p)$']]
        tb = tb.loc[[x for x in tb.index if x not in ['$R^2$,$R^2_{btw}$,$R^2_{wthn}$', 'loglik, $LB_{10}(p)$']]]
        tb.sort_index(inplace=True)
        tb = tb.append(test_stats)
        tb = tb.reset_index(drop=False)

        for i, x in enumerate(tb[fn]):
            if x in list(rowname_map.keys()):
                tb[fn][i] = rowname_map[x]

        # nicer
        newInd = tb[fn].str.split('.', expand=True)
        newInd.columns = ['Type', 'Var.Name']
        tb = tb.drop([fn], axis=1)
        tb = tb.set_index(pd.MultiIndex.from_frame(newInd))



        tb.to_latex(os.path.expanduser('~/Downloads/DeepArticle/tables/') + 'test'+fn+str(k)+'_2023.tex', escape = False, index = True)


