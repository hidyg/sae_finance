###################################v##########################################
###################################v##########################################
# univariate analysis monthly
###################################v##########################################
###################################v##########################################
require(xts)
require(midasr)
require(data.table)
require(xtable)

global_start = as.Date('2004-01-01')
global_ends = c(as.Date("2021-12-31"),as.Date("2019-12-31") ) # 2039-12-31
# train / test
train_end = as.Date("2010-12-31")   #
test_start = as.Date(train_end+1)

# lags
lags_in = 1


# I) GDP: Y
cntrs = c('US', 'EU', 'AU', 'GB', 'CA', 'JP')

# select cntr
c = 'US'  # country

global_end = as.Date(global_end[1])
Y = read.csv(file = paste0('~/Documents/BareBoneBT/data/workspace/midasY_',c,'.csv'),sep=",")
Y[,1] = as.Date(Y[,1]); Y = Y[Y[,1]>= global_start & Y[,1]<= global_end,]


# II) individual macro variables
# z scored versions US, EU, etc.
X = read.csv(file = paste0('~/Documents/BareBoneBT/data/workspace/midasX_',c,'_zscored.csv'), sep=",")

xmacr = xts(X[,2:NCOL(X)], as.Date(X[,1]))
# kill NAs
xmacr = na.locf(xmacr, na.rm = F)
# get last observation per month
xmacr = apply.monthly(xmacr, tail,1)



# map to mth ends
month_ends = seq(as.Date("2002-01-01"),as.Date("2022-01-01"),by="months")-1
xts_mthend = xts(rep(1,length(month_ends)),month_ends)
# map macro series to month ends
xmacr = xmacr[index(xmacr)>"2001-01-01",]
xmacr = merge(xmacr, xts_mthend, all = c(T,T) )  # get month ends
xmacr = na.locf(xmacr)
xmacr = xmacr[index(xmacr) %in% index(xts_mthend),1:(NCOL(xmacr)-1)] # drop col of ones
xmacr = na.fill(xmacr, fill = list(0,0,0)) # remaining nan fill

xmacr = xmacr[index(xmacr)>=global_start & index(xmacr)<=global_end,]


# III) sentiment representations
# dimension of representation
dim_rep = 5
Xs = read.csv(file = paste0('~/Documents/BareBoneBT/data/workspace/',c,'repr_2023.csv'), sep=",")
Xs[,1] = as.Date(Xs[,1])
# group by month and aggregate
xrep_xts = xts(Xs[,2:NCOL(Xs)],as.Date(Xs[,1]))
for (colmn in names(xrep_xts)) {
  xrep_xts_sub = xrep_xts[,colmn]
  if (colmn==names(xrep_xts)[1]) {
    xrep = apply.monthly(xrep_xts_sub, FUN=function(x) mean(x, na.rm=T))
  } else {
    xrep_m = apply.monthly(xrep_xts_sub, FUN=function(x) mean(x, na.rm=T))
    xrep = merge(xrep,xrep_m, all = c(T,T) )
  }
}
xrep = xrep[index(xrep)>global_start & index(xrep)<=global_end,]


# IV) macro vars
# to xts
yy = xts(c(Y[,2]),c(as.Date(Y[,1]) ) )
xmacrx = xmacr
# handle xrep xts:
xrepx = xrep

# set end input
end_in = global_end



# V) analysis
# cutoff is 2004-01-01
xmacr = window(xmacrx, start = global_start, end = end_in )
# same for representations
xrep = window(xrepx, start = global_start, end = end_in )
y = window(yy, start = global_start, end = end_in )
# attach freq.
y = ts(y, start = c(2004,01), frequency =  4)

# list of Xs for macro
lmacr = list(y = y)
for (i in seq(1,NCOL(xmacr))) {
  # to list
  lmacr[[names(xmacr)[i]]] = ts( xmacrx[,i], start = c(2004, 01), frequency = 12 )
  # construct all variables in workspace
  eval(parse(text=paste(names(xmacr)[i],'=ts( xmacr[,i], start = c(2004, 01), frequency = 12 )',sep=""))) # '=lmacr[[names(xmacr)[i]]]'
}
lrep = list(y = y)
for (i in seq(1,NCOL(xrep))) {
  # to list
  lrep[[names(xrep)[i]]] = ts( xrepx[,i], start = c(2004, 01), frequency = 12 )
  # construct all variables in workspace
  eval(parse(text=paste(names(xrep)[i],'=ts( xrep[,i], start = c(2004, 01), frequency = 12 )',sep=""))) # '=lrep[[names(xrep)[i]]]'
}

# needed: random walk benchmark  ( y ~ mls(y, 1, 1) )

# formulas are given as y ~ mls(y, 1, 1)
#   mls( x, k, m)    k is the (HF) lag vector... e.g. 3:12
#  m is the frequency ratio: for each low frequency period we observe the HF process m times
y_bm = window(yy, start = global_start, end = train_end )
# attach freq.
y_bm = ts(y_bm, start = c(2004,01), frequency =  4)
# random walk benchmark
rm_bm = midas_r( y_bm ~ mls(y_bm, 1, 1), start=list(y_bm=c(0.3) ))
y_bm_test = window(yy, start = test_start, end = global_end )
# attach freq.
y_bm_test = ts(y_bm_test, start = c(2010,01), frequency =  4)

rm_bm_pred = predict( rm_bm, newdata = list(y_bm = y_bm_test ) )
bm_MSE = mean((rm_bm_pred - y_bm_test[2:length(y_bm_test)])^2)

# random walk
bm_RW_MSE = mean((lag.xts(y_bm_test) - y_bm_test)^2,na.rm=T)

##############################################################################
# single macro vars
##############################################################################
# construct formula
xselection =  c(seq(1,NCOL(xmacr))) #seq(1,10)#c(1,3)
mls_fct = ',nealmon' #',nealmon' # ,nbetaMT  # ,nealmon # UMIDAS: ' '
startvalues = 'c(0.3,0.3)'
df_mse = data.frame(matrix(0,3,length(xselection))); names(df_mse) = names(xmacr)[xselection]
fitted_list = list()

# macro variable comparison
for (i in xselection) {
  print(names(xmacr)[i])
  xstr = ' '
  startstr = 'list('
  xstr = paste(xstr, paste('+ mls(',names(xmacr)[i],', ',toString(lags_in),':11, 3',
                           mls_fct,')',sep=""),sep="" )

  startstr = paste(startstr,names(xmacr)[i],'=',startvalues,' ) ',sep="")
  # could be in loop to get model fits for several variables
  f_str = paste('midas_r( y ~ mls(y, 1, 1) ',xstr,',start=',startstr,')',sep="")
  startstr = startstr #  UMIDAS: 'NULL'
  fitted_model <- eval(parse(text = f_str ) )
  fm_summary = summary(fitted_model)

  # MSE
  mse_model = mean(fitted_model$residuals^2)
  # over- underperf vs. BM
  df_mse[1,names(xmacr)[i]] = mse_model/bm_MSE
  df_mse[2,names(xmacr)[i]] = fm_summary$r_squared
  df_mse[3,names(xmacr)[i]] = fm_summary$adj_r_squared

  # save fitted model
  fitted_list[[names(xmacr)[i] ]] = fitted_model
}

###################################v##########################################
# add sentiment representations
###################################v##########################################
# construct formula
xselection =  c(seq(1,NCOL(xrep))) #seq(1,10)#c(1,3)
mls_fct = ',nealmon' #',nealmon' # ,nbetaMT  # ,nealmon # UMIDAS: ' '
startvalues = 'c(0.3,0.3)'
df_mse_rep = data.frame(matrix(0,3,length(xselection))); names(df_mse_rep) = names(xrep)[xselection]
fitted_list_rep = list()

# sent variable comparison
for (i in xselection) {
  print(names(xrep)[i])
  xstr = ' '
  startstr = 'list('
  xstr = paste(xstr, paste('+ mls(',names(xrep)[i],', ',toString(lags_in),':11, 3',
                           mls_fct,')',sep=""),sep="" )
  #
  startstr = paste(startstr,names(xrep)[i],'=',startvalues,' ) ',sep="")
  # could be in loop to get model fits for several variables
  f_str = paste('midas_r( y ~ mls(y, 1, 1) ',xstr,',start=',startstr,')',sep="")
  startstr = startstr #  UMIDAS: 'NULL'
  fitted_model <- eval(parse(text = f_str ) )
  fm_summary = summary(fitted_model)

  # MSE
  mse_model = mean(fitted_model$residuals^2)
  # over- underperf vs. BM
  df_mse_rep[1,names(xrep)[i]] = mse_model/bm_MSE
  df_mse_rep[2,names(xrep)[i]] = fm_summary$r_squared
  df_mse_rep[3,names(xrep)[i]] = fm_summary$adj_r_squared

  # save fitted model
  fitted_list_rep[[names(xrep)[i] ]] = fitted_model
}

########################################
#########  OOS analysis ################
########################################
# set up names
names_df = unlist(lapply(matrix(names(xrepx),dim(xrepx)[2],1),
                         FUN = function(x) {strsplit(x, 'H')[[1]][1]}))
# handle PCA names
names_df[1:(dim_rep*2)] = unlist(lapply(matrix(names_df[1:(dim_rep*2)],(dim_rep*2),1),
                                        FUN = function(x) { substr(x, 1, (nchar(x)-1) ) }))
names_df = unique(names_df)


# result collection
# av mse df
df_avmse =  data.frame(matrix(0,5,(length(names_df)+2))); names(df_avmse) = c(names_df,"macro","combo")
l_fc_errors = list()
l_fc = list()
l_y = list()

for (k in seq(1,length(names_df))) {
  # from matlab toolbox:
  # slots are
  #%      'MSFE':  weights by mean squared error of forecast (default)
  #%      'DMSFE': weights by discounted mean squared error of forecast !!!!-> used in publication
  #%      'aic':   weights by Akaike information criteria of the regression
  #%      'bic':   weights by Bayesian information criteria of the regression
  #%      'Flat':  equal weights
  models_list = fitted_list_rep[((k-1)*dim_rep+1):(k*dim_rep)]
  data_in = lrep[c("y",names(models_list))]
  insample = 1:which(index(yy) == train_end)
  outsample = (which(index(yy) == train_end)+1):length(lrep$y)
  avgf = average_forecast(models_list, data = data_in,insample = insample, outsample = outsample)
  l_fc_errors[[k]] = avgf$avgforecast[,"DMSFE"] - data_in$y[outsample]; l_fc[[k]] = avgf$avgforecast[,"DMSFE"]
  l_y[[k]] = data_in$y[outsample]
  plot(data_in$y[outsample], type = "b", col = "red", main = names_df[k]);lines(avgf$avgforecast[,"DMSFE"], type = "l");
  df_avmse[1,names_df[k]] = avgf$accuracy$average$MSE[avgf$accuracy$average$Scheme=="DMSFE"] #  squared discounted mean squared forecast error (MSFE) combinations
  df_avmse[2,names_df[k]] = avgf$accuracy$average$MSE[avgf$accuracy$average$Scheme=="DMSFE"]/bm_MSE
  df_avmse[3,names_df[k]] = avgf$accuracy$average$MSE[avgf$accuracy$average$Scheme=="DMSFE"]/bm_RW_MSE
  df_avmse[4,names_df[k]] = avgf$accuracy$average$MSE[avgf$accuracy$average$Scheme=="EW"]/bm_MSE
  df_avmse[5,names_df[k]] = avgf$accuracy$average$MSE[avgf$accuracy$average$Scheme=="EW"]/bm_RW_MSE
}

