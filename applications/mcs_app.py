import pandas as pd
from arch.bootstrap import MCS
import pickle as pckl

# 1) TSM
# check: is that the equal weighted strategy?  -> eq only?
stratl = pd.read_csv( 'stratreturns_long.csv')
strats = pd.read_csv( 'stratreturns_short.csv')

# loss is (-1) * (strt subtract passive) -> high underperf -> high loss
NNname_map =  {'Fs': 'ReturnF',
 'idx': 'idx',
 'Passive': 'Passive',
 'NN': 'supervised AE',
 'NNPureClassif_d_': 'NN Class.',
 'NNPLSreg_d_': 'PLS',
 'NNGRUAE_d_': 'GRU',
 'NNPrCA_d_': 'PCA'}

# vs passive
stratl.rename({c:NNname_map[c] for c in NNname_map.keys()}, axis =1, inplace = True )
strats.rename({c:NNname_map[c] for c in NNname_map.keys()}, axis =1, inplace = True )


stratl = stratl[['supervised AE','NN Class.','PLS','GRU','PCA']]
strats = strats[['supervised AE','NN Class.','PLS','GRU','PCA']]

# p-values short
mcs = MCS(-strats, size=0.05)
mcs.compute()
print("MCS P-values")
print(mcs.pvalues)
# p-values long
mcs = MCS(-stratl, size=0.05)
mcs.compute()
print("MCS P-values")
print(mcs.pvalues)



# 2) GDP forecasting
# use the following files ... per country
cntrs = ['US', 'EU', 'AU', 'GB', 'CA', 'JP']
fls =  [ ('~/Documents/MarketRepR/csv_data/fcerr_m_'+c+'.csv',
          '~/Documents/MarketRepR/csv_data/fcerr_d_'+c+'.csv',
          '~/Documents/MarketRepR/csv_data/fcerr_m_exCvd_'+c+'.csv',
          '~/Documents/MarketRepR/csv_data/fcerr_d_exCvd_'+c+'.csv') for c in cntrs]


# headers "Supervised AE","NN Classifier","GRUAE","PLS","PCA (sentiment)","PCA (Macro Var.)"
for i,c in enumerate(cntrs):

    m = pd.read_csv('~/Documents/MarketRepR/csv_data/fcerr_m_'+c+'.csv',index_col=0)**2
    mxC = pd.read_csv('~/Documents/MarketRepR/csv_data/fcerr_m_exCvd_'+c+'.csv',index_col=0)**2
    d = pd.read_csv('~/Documents/MarketRepR/csv_data/fcerr_d_'+c+'.csv',index_col=0)**2
    dxC = pd.read_csv('~/Documents/MarketRepR/csv_data/fcerr_d_exCvd_'+c+'.csv',index_col=0)**2

    if c in ['EU','US']:
        mcs = MCS(m, size=0.05)
        mcs.compute()
        print("MCS P-values m for country "+c)
        print(mcs.pvalues)
        mcs = MCS(mxC, size=0.05)
        mcs.compute()
        print("MCS P-values mxC for country " + c)
        print(mcs.pvalues)
        mcs = MCS(d, size=0.05)
        mcs.compute()
        print("MCS P-values d for country " + c)
        print(mcs.pvalues)
        mcs = MCS(dxC, size=0.05)
        mcs.compute()
        print("MCS P-values dxC for country " + c)
        print(mcs.pvalues)

    if i==0:
        m_all,mxC_all,d_all,dxC_all = m,mxC,d,dxC
    else:
        m_all = m_all.add( m)
        mxC_all = mxC_all.add(mxC_all)
        d_all = d_all.add(d_all)
        dxC_all = dxC_all.add(dxC_all)

# ALL CNTRS
mcs = MCS(m_all, size=0.05)
mcs.compute()
print("MCS P-values m for country ")
print(mcs.pvalues)
mcs = MCS(mxC_all, size=0.05)
mcs.compute()
print("MCS P-values mxC for country ")
print(mcs.pvalues)
mcs = MCS(d_all, size=0.05)
mcs.compute()
print("MCS P-values d for country ")
print(mcs.pvalues)
mcs = MCS(dxC_all, size=0.05)
mcs.compute()
print("MCS P-values dxC for country ")
print(mcs.pvalues)


# 3) the panel
keys_sel = ['d:SupAE',
 'd:NNclssfr',
 'k:GRU',
 'p:PCAroll',
 'p:PLSregr',
 'q:Lasso',
 'r:logregL1']

# single files residuals of panel in resids.pckl
with open('resids	ext{FX: }r_{.pckl','rb') as f:
    resids = pckl.load(f)

pd_r = pd.DataFrame(resids)
# squared error loss
mcs = MCS(pd_r**2, size=0.05)
mcs.compute()
print("MCS P-values residuals panel ")
print(mcs.pvalues)