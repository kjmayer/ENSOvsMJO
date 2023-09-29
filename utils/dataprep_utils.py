#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
@author: Emily Gordon
@editor: Kirsten Mayer
"""
import xarray as xr
import numpy as np
import sys
sys.path.append('/glade/work/kjmayer/research/catalyst/ENSOvsMJO/utils/')
# sys.path.append('/glade/u/home/wchapman/ENSOvsMJO/utils/')
from exp_hp import get_hp
import random
import tensorflow as tf
from datetime import datetime
import pandas as pd


# data prep functions
def subset(x, y, n_valzero, n_valone, i_valzero, i_valone):
    # randomly subset predictand data for equal classes
    if n_valone != n_valzero:
        if n_valone > n_valzero:
            isubset_valone = np.random.choice(i_valone,size=n_valzero,replace=False)
            i_valnew = np.sort(np.append(i_valzero,isubset_valone))

        elif n_valone < n_valzero:
            isubset_valzero = np.random.choice(i_valzero,size=n_valone,replace=False)
            i_valnew = np.sort(np.append(isubset_valzero,i_valone))

        y  = y.isel(time = i_valnew,drop=True) 
        x  = x.isel(time = i_valnew,drop=True)
    return x,y,i_valnew

def days_in_year(Ytrain):
    _DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]  # Ignores leap days.
    start_year, end_year = int(Ytrain.time.dt.year[0]), int(Ytrain.time.dt.year[-1])+1
    dates = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            for day in range(1, _DAYS_IN_MONTH[month-1]+1):
                dates.append(datetime(year, month, day, hour=0, minute=0))
                
    dates = pd.to_datetime(dates)
    dates_a2a = dates[119:(36500+119)]
    return dates,dates_a2a


def get_testing(N_z500runmean,LEAD):
    DIR = '/glade/scratch/kjmayer/DATA/CESM2-piControl/daily/'
    X1_FINAME = 'SSTv3_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc'
    X2_FINAME = 'MJO_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc'
    Y_FINAME  = 'Z500v2_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc'

    N_days_100yrs = 100*365 #days
    x_months = [11,12,1,2]
    
    EXP_NAME = 'default'
    hps = get_hp(EXP_NAME)
    
    X1VAR  = hps['INPUT'][0] #'TS_SST_ONI'
    X2VAR1 = hps['INPUT'][1] #'RMM1_CESM2'
    X2VAR2 = hps['INPUT'][2] #'RMM2_CESM2'
    YVAR   = hps['OUTPUT'][0]#'TS_Z500a'

    N_daysbefore = hps['X_ADDITIONAL_DAYS']
    GLOBAL_SEED = hps['GLOBAL_SEED']
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    tf.random.set_seed(GLOBAL_SEED)
    
    
    # ---------- TRAINING (100 years) ----------
    # predictors [time]
    X1train = xr.open_dataset(DIR+X1_FINAME)[X1VAR][:N_days_100yrs]

    X2trainRMM1 = xr.open_dataset(DIR+X2_FINAME)[X2VAR1][:N_days_100yrs]
    X2trainRMM2 = xr.open_dataset(DIR+X2_FINAME)[X2VAR2][:N_days_100yrs]
    X2train = xr.concat([X2trainRMM1,X2trainRMM2], dim = 'new_dim') # 2xtime
    del X2trainRMM1, X2trainRMM2

    # predictand [time]
    Ytrain = xr.open_dataset(DIR+Y_FINAME)[YVAR][:N_days_100yrs]

    
    
    # ---------- TESTING (100 years) ----------
    # repeat for validation data
    X1test = xr.open_dataset(DIR+X1_FINAME)[X1VAR][N_days_100yrs*2:]

    X2testRMM1 = xr.open_dataset(DIR+X2_FINAME)[X2VAR1][N_days_100yrs*2:]
    X2testRMM2 = xr.open_dataset(DIR+X2_FINAME)[X2VAR2][N_days_100yrs*2:]
    X2test = xr.concat([X2testRMM1,X2testRMM2], dim = 'new_dim') # 2xtime

    # predictand [time]
    Ytest = xr.open_dataset(DIR+Y_FINAME)[YVAR][N_days_100yrs*2:]
    
    
    
    # ---------- running mean ----------
    dates, dates_a2a = days_in_year(Ytrain) # dates april to april & ignore leap day

    Ytrainroll = Ytrain.rolling(time=N_z500runmean, min_periods=N_z500runmean, center=False).mean('time').dropna('time')
    Ytrainroll['time'] = dates_a2a[:-(N_z500runmean-1)]

    # ----------
    dates, dates_a2a = days_in_year(Ytest) # dates april to april & ignore leap day
    Ytestroll = Ytest.rolling(time=N_z500runmean, min_periods=N_z500runmean, center=False).mean('time').dropna('time')
    Ytestroll['time'] = dates_a2a[:-(N_z500runmean-1)]
    
    
    
    # ----------
    # end X early, so we don't run out of Y data
    X1train_final = X1train[:-(N_z500runmean-1)][:-1*LEAD] 
    X2train_final = X2train[:,:-(N_z500runmean-1)][:,:-1*LEAD] 
    Ytrain_final = Ytrainroll[LEAD:]

    # repeat for testing data
    X1test_final = X1test[:-(N_z500runmean-1)][:-1*LEAD] 
    X2test_final = X2test[:,:-(N_z500runmean-1)][:,:-1*LEAD] 
    Ytest_final = Ytestroll[LEAD:]
    
    # ---------- standardization values ----------
    X1mean = X1train_final.groupby('time.dayofyear').mean()
    X1std = X1train_final.groupby('time.dayofyear').std()
    X2mean = X2train_final.groupby('time.dayofyear').mean()
    X2std = X2train_final.groupby('time.dayofyear').std()
    
    Ymed = Ytrain_final.quantile(q=.5,dim='time')

    

    # ---------- TESTING STANDARDIZATION ----------
    X1test_norm = (X1test_final.groupby('time.dayofyear')- X1mean).groupby('time.dayofyear')/(X1std)
    X2test_norm = (X2test_final.groupby('time.dayofyear')- X2mean).groupby('time.dayofyear')/(X2std)

    Ytest_norm = Ytest_final - Ymed

    # turn Ys into 0s and 1s
    Ytest_norm[Ytest_norm<=0] = 0 
    Ytest_norm[Ytest_norm>0] = 1

    # convert data from xarray to numpy 
    Xtest_time = X1test_norm.time
    X1_test = X1test_norm.T.values
    X2_test = X2test_norm.T.values

    Ytest_time = Ytest_norm.time
    Y_test = Ytest_norm.values
    
    
    
    # ---------- add memory ----------
    for ee,num in enumerate(reversed(range(N_daysbefore + 1))):
        X1_test_back = X1_test[num:-(ee+1)] 
        X2_test_back = X2_test[num:-(ee+1),:]
        if ee==0:
            X1_test_norm_mem=X1_test[num:-(ee+1)]
            X2_test_norm_mem=X2_test[num:-(ee+1),:]
        else: 
            X1_test_norm_mem = np.vstack([X1_test_back,X1_test_norm_mem])
            X2_test_norm_mem = np.concatenate([X2_test_back,X2_test_norm_mem],axis=1)

    Ytest_time_mem = Ytest_time[:-(N_daysbefore+1)]      
    Y_test_mem=Y_test[:-(N_daysbefore+1)] #adjust Ytarget....

    Xtest_time_mem = Xtest_time[:-(N_daysbefore+1)]
    X1_test_norm_mem = X1_test_norm_mem.T

    # ---------- convert to xarray 
    X1_testxr_mem = xr.DataArray(data=X1_test_norm_mem,
                            dims=["time","memory"],
                            coords={'time':Xtest_time_mem, 'memory':np.arange(0,N_daysbefore+1)})

    X2_testxr_mem = xr.DataArray(data=X2_test_norm_mem,
                            dims=["time","memoryx2"],
                            coords={'time':Xtest_time_mem, 'memoryx2':np.arange(0,(N_daysbefore+1)*2)})

    Y_testxr_mem = xr.DataArray(data=Y_test_mem,
                            dims=["time"],
                            coords={'time':Ytest_time_mem})

    

    # ---------- get NDJF(M) ----------
    itest_xndjf = np.where(X1_testxr_mem.time.dt.month.isin(x_months))

    X1_testxr_mem_NDJF = X1_testxr_mem[X1_testxr_mem.time.dt.month.isin(x_months)]
    X2_testxr_mem_NDJF = X2_testxr_mem[X2_testxr_mem.time.dt.month.isin(x_months)]

    Y_testxr_mem_NDJFM = Y_testxr_mem[itest_xndjf]
    

    
    # ---------- subset predictand (and predictors) to same number of 0s and 1s ----------
    n_testzero = np.shape(np.where(Y_testxr_mem_NDJFM==0)[0])[0]
    n_testone  = np.shape(np.where(Y_testxr_mem_NDJFM==1)[0])[0]
    i_testzero = np.where(Y_testxr_mem_NDJFM==0)[0]
    i_testone  = np.where(Y_testxr_mem_NDJFM==1)[0]
    X1_testxr_mem_NDJF, Y_testxr_mem_NDJFM, i_testnew = subset(X1_testxr_mem_NDJF, Y_testxr_mem_NDJFM, n_testzero, n_testone, i_testzero, i_testone)
    X2_testxr_mem_NDJF = X2_testxr_mem_NDJF.isel(time = i_testnew,drop=True)


    return X1_testxr_mem_NDJF, X2_testxr_mem_NDJF, Y_testxr_mem_NDJFM


# 
def get_testing_obs(N_z500runmean,LEAD):
    DIR = '/glade/scratch/kjmayer/DATA/CESM2-piControl/daily/'

    N_days_100yrs = 100*365 #days
    x_months = [11,12,1,2]
    
    EXP_NAME = 'default'
    hps = get_hp(EXP_NAME)
    
    X1VAR  = hps['INPUT'][0] #'TS_SST_ONI_45'
    X2VAR1 = hps['INPUT'][1] #'RMM1_CESM2'
    X2VAR2 = hps['INPUT'][2] #'RMM2_CESM2'
    YVAR   = hps['OUTPUT'][0]#'TS_Z500a'

    N_daysbefore = hps['X_ADDITIONAL_DAYS']
    GLOBAL_SEED = hps['GLOBAL_SEED']
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    tf.random.set_seed(GLOBAL_SEED)
    
    
    # ---------- TRAINING (100 years) ----------
    X1_FINAME = 'SSTv3_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc'
    X2_FINAME = 'MJO_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc'
    Y_FINAME  = 'Z500v2_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc'
    
    # predictors [time]
    X1train = xr.open_dataset(DIR+X1_FINAME)[X1VAR][:N_days_100yrs]

    X2trainRMM1 = xr.open_dataset(DIR+X2_FINAME)[X2VAR1][:N_days_100yrs]
    X2trainRMM2 = xr.open_dataset(DIR+X2_FINAME)[X2VAR2][:N_days_100yrs]
    X2train = xr.concat([X2trainRMM1,X2trainRMM2], dim = 'new_dim') # 2xtime
    del X2trainRMM1, X2trainRMM2

    # predictand [time]
    Ytrain = xr.open_dataset(DIR+Y_FINAME)[YVAR][:N_days_100yrs]

    
    # ---------- TESTING (1974-2020) ----------
    X1_OBS_FINAME = 'SSTv2_ERA5_1974_2022.001.nc'
    X2_OBS_FINAME = 'Observed_MJO_BOMindex.nc'
    Y_OBS_FINAME  = 'Z500v2_ERA5_1974_2020.001.nc'
    
    # repeat for testing data
    X2testRMM1 = xr.open_dataset(DIR+X2_OBS_FINAME)['RMM1_obs_BOM']
    X2testRMM2 = xr.open_dataset(DIR+X2_OBS_FINAME)['RMM2_obs_BOM']
    X2test = xr.concat([X2testRMM1,X2testRMM2], dim = 'new_dim') # 2xtime

    X1test = xr.open_dataset(DIR+X1_OBS_FINAME)[X1VAR][int(X2testRMM1.time.dt.dayofyear[0])-1:]

    # predictand [time]
    Ytest = xr.open_dataset(DIR+Y_OBS_FINAME)[YVAR][int(X2testRMM1.time.dt.dayofyear[0])-1:].squeeze()
    
    
    
    # ---------- running mean ----------
    dates, dates_a2a = days_in_year(Ytrain)

    Ytrainroll = Ytrain.rolling(time=N_z500runmean, min_periods=N_z500runmean, center=False).mean('time').dropna('time')
    Ytrainroll['time'] = dates_a2a[:-(N_z500runmean-1)]

    # ----------
    Ytestroll = Ytest.rolling(time=N_z500runmean, min_periods=N_z500runmean, center=False).mean('time').dropna('time')
    Ytestroll['time'] = Ytest['time'][:-(N_z500runmean-1)]
    
    
    
    # ----------
    # end X early, so we don't run out of Y data
    X1train_final = X1train[:-(N_z500runmean-1)][:-1*LEAD] 
    X2train_final = X2train[:,:-(N_z500runmean-1)][:,:-1*LEAD] 
    Ytrain_final = Ytrainroll[LEAD:]

    # repeat for testing data
    X1test_final = X1test[:-(N_z500runmean-1)][:-1*LEAD] 
    X2test_final = X2test[:,:-(N_z500runmean-1)][:,:-1*LEAD] 
    Ytest_final = Ytestroll[LEAD:]
    
    # ---------- standardization values ----------
    X1mean = X1train_final.groupby('time.dayofyear').mean()
    X1std = X1train_final.groupby('time.dayofyear').std()
    X2mean = X2train_final.groupby('time.dayofyear').mean()
    X2std = X2train_final.groupby('time.dayofyear').std()
    
    Ymed = Ytrain_final.quantile(q=.5,dim='time')

    

    # ---------- TESTING STANDARDIZATION ----------
    X1test_norm = (X1test_final.groupby('time.dayofyear')- X1mean).groupby('time.dayofyear')/(X1std)
    X2test_norm = (X2test_final.groupby('time.dayofyear')- X2mean).groupby('time.dayofyear')/(X2std)

    Ytest_norm = Ytest_final - Ymed

    # turn Ys into 0s and 1s
    Ytest_norm[Ytest_norm<=0] = 0 
    Ytest_norm[Ytest_norm>0] = 1

    # convert data from xarray to numpy 
    Xtest_time = X1test_norm.time
    X1_test = X1test_norm.T.values
    X2_test = X2test_norm.T.values

    Ytest_time = Ytest_norm.time
    Y_test = Ytest_norm.values
    
    
    
    # ---------- add memory ----------
    for ee,num in enumerate(reversed(range(N_daysbefore + 1))):
        X1_test_back = X1_test[num:-(ee+1)] 
        X2_test_back = X2_test[num:-(ee+1),:]
        if ee==0:
            X1_test_norm_mem=X1_test[num:-(ee+1)]
            X2_test_norm_mem=X2_test[num:-(ee+1),:]
        else: 
            X1_test_norm_mem = np.vstack([X1_test_back,X1_test_norm_mem])
            X2_test_norm_mem = np.concatenate([X2_test_back,X2_test_norm_mem],axis=1)

    Ytest_time_mem = Ytest_time[:-(N_daysbefore+1)]      
    Y_test_mem=Y_test[:-(N_daysbefore+1)] #adjust Ytarget....

    Xtest_time_mem = Xtest_time[:-(N_daysbefore+1)]
    X1_test_norm_mem = X1_test_norm_mem.T

    # ---------- convert to xarray 
    X1_testxr_mem = xr.DataArray(data=X1_test_norm_mem,
                            dims=["time","memory"],
                            coords={'time':Xtest_time_mem, 'memory':np.arange(0,N_daysbefore+1)})

    X2_testxr_mem = xr.DataArray(data=X2_test_norm_mem,
                            dims=["time","memoryx2"],
                            coords={'time':Xtest_time_mem, 'memoryx2':np.arange(0,(N_daysbefore+1)*2)})

    Y_testxr_mem = xr.DataArray(data=Y_test_mem,
                            dims=["time"],
                            coords={'time':Ytest_time_mem})

    

    # ---------- get NDJF(M) ----------
    itest_xndjf = np.where(X1_testxr_mem.time.dt.month.isin(x_months))

    X1_testxr_mem_NDJF = X1_testxr_mem[X1_testxr_mem.time.dt.month.isin(x_months)]
    X2_testxr_mem_NDJF = X2_testxr_mem[X2_testxr_mem.time.dt.month.isin(x_months)]

    Y_testxr_mem_NDJFM = Y_testxr_mem[itest_xndjf]
    

    
    # ---------- subset predictand (and predictors) to same number of 0s and 1s ----------
    n_testzero = np.shape(np.where(Y_testxr_mem_NDJFM==0)[0])[0]
    n_testone  = np.shape(np.where(Y_testxr_mem_NDJFM==1)[0])[0]
    i_testzero = np.where(Y_testxr_mem_NDJFM==0)[0]
    i_testone  = np.where(Y_testxr_mem_NDJFM==1)[0]
    X1_testxr_mem_NDJF, Y_testxr_mem_NDJFM, i_testnew = subset(X1_testxr_mem_NDJF, Y_testxr_mem_NDJFM, n_testzero, n_testone, i_testzero, i_testone)
    X2_testxr_mem_NDJF = X2_testxr_mem_NDJF.isel(time = i_testnew,drop=True)


    return X1_testxr_mem_NDJF, X2_testxr_mem_NDJF, Y_testxr_mem_NDJFM
