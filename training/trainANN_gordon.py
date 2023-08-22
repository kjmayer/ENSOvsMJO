"""
Authors: wchapman@ucar.edu / kjmayer@ucar.edu
Will Chapman & Kirsten Mayer 
"""

import argparse
import json
import os
import time
import copy
import warnings
import numpy as np
import sys
sys.path.append('/glade/work/kjmayer/research/catalyst/ENSOvsMJO/utils/')
import ast


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import random
from sklearn.metrics import accuracy_score

import tensorflow as tf
from trainGordon_utils import subset, build_model, fullmodel, scheduler, plot_results
from exp_hp import get_hp


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

def parse_list(string):
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("Invalid list format")

if __name__ == '__main__':
    print('...starting to fly captain...')
    parser = argparse.ArgumentParser(description='MJOvENSO')
    parser.add_argument("--GLOBAL_SEED", default=32, type=int, help='provide a seed to train with')
    parser.add_argument("--Y_RUNMEAN", default=7, type=int, help='provide a seed to train with')
    parser.add_argument("--X_ADDITIONAL_DAYS",default=10,type=int, help='provide a seed to train with')
    parser.add_argument("--LEAD",default=14,type=int, help='provide a seed to train with')
    parser.add_argument("--EXP_NAME",default='exp1',type=str, help='provide a seed to train with')

    parser.add_argument("--DROPOUT_RATE",default=0,type=int, help='provide a seed to train with')
    parser.add_argument("--RIDGE1",default=0,type=int, help='provide a seed to train with')
    parser.add_argument("--RIDGE2",default=0,type=int, help='provide a seed to train with')
    parser.add_argument("--HIDDENS2",default=[8],type=parse_list, help='provide a seed to train with')
    parser.add_argument("--HIDDENS1",default=[8],type=parse_list, help='provide a seed to train with')
    parser.add_argument("--BATCH_SIZE",default=32,type=int, help='provide a seed to train with')
    parser.add_argument("--PATIENCE",default=20,type=int, help='provide a seed to train with')
    parser.add_argument("--LR",default=0.001,type=int, help='provide a seed to train with')
    parser.add_argument("--SEED",default=1,type=int, help='provide a seed to train with')

    
    args = parser.parse_args()
    
    params={'XVARS':['TS_SST_ONI','RMM1_CESM2','RMM2_CESM2'], 
        'YVAR':['TS_Z500a'],
        'DIR':'/glade/scratch/kjmayer/DATA/CESM2-piControl/daily/', 
        'X1_FINAME':'SSTv2_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc',
        'X2_FINAME':'MJO_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc',
        'Y_FINAME':'Z500v2_CESM2_0100_0400.b.e21.B1850.f09_g17.CMIP6-esm-piControl.001.nc',
        'EXP_NAME':'exp1',
        'model_dir':'/glade/work/kjmayer/research/catalyst/ENSOvsMJO/saved_models/',
        'batchsize':64,
        'BASEDIR':'/glade/work/wchapman/DA_ML/CESML_AI/',
        'loss':'MSE',
        'optimizer':"adam",
        'lr':0.005,
        'shuffle':True,
        'epochs':28,
        'seed':40,
        'model_dir': "/glade/work/kjmayer/research/catalyst/ENSOvsMJO/saved_models/",
        'model': 'ANN_MSE',
        'SEED':1,
         }
    
    EXP_NAME = args.EXP_NAME or params['EXP_NAME']
    hps = get_hp(EXP_NAME)
    params['SEED'] = args.SEED or params['SEED']
    params['DROPOUT_RATE'] = args.DROPOUT_RATE or hps['DROPOUT_RATE']
    params['RIDGE1'] = args.RIDGE1 or hps['RIDGE1']
    params['RIDGE2'] = args.RIDGE2 or hps['RIDGE2']
    params['HIDDENS1'] = args.HIDDENS1 or hps['HIDDENS1']
    params['HIDDENS2'] = args.HIDDENS2 or hps['HIDDENS2']
    params['BATCH_SIZE'] = args.BATCH_SIZE or hps['BATCH_SIZE']
    params['PATIENCE'] = args.PATIENCE or hps['PATIENCE']
    params['LR'] = args.LR or hps['LR']
    params['GLOBAL_SEED'] = args.GLOBAL_SEED or hps['GLOBAL_SEED']
    params['Y_RUNMEAN'] = args.Y_RUNMEAN or hps['Y_RUNMEAN']
    params['X_ADDITIONAL_DAYS'] = args.X_ADDITIONAL_DAYS or hps['X_ADDITIONAL_DAYS']
    params['LEAD'] = args.LEAD or hps['LEAD']
    
    print(params)
    
    np.random.seed(params['GLOBAL_SEED'])
    random.seed( params['GLOBAL_SEED'])
    tf.random.set_seed( params['GLOBAL_SEED'])
    
    
    #get data:
    N_z500runmean = params['Y_RUNMEAN'] #days
    N_days_100yrs = 100*365 #days
    LEAD = params['LEAD'] #days
    N_daysbefore =params['X_ADDITIONAL_DAYS'] #days to go "back in time" for X
    
    # TRAINING (100 years)
    # predictors [time]
    DIR = params['DIR']
    X1_FINAME = params['X1_FINAME']
    X2_FINAME = params['X2_FINAME']
    Y_FINAME = params['Y_FINAME']
    YVAR = params['YVAR'][0]
    X1VAR = params['XVARS'][0]
    X2VAR1 = params['XVARS'][1]
    X2VAR2 = params['XVARS'][2]
    
    X1train = xr.open_dataset(DIR+X1_FINAME)[X1VAR][:N_days_100yrs]
    X2trainRMM1 = xr.open_dataset(DIR+X2_FINAME)[X2VAR1][:N_days_100yrs]
    X2trainRMM2 = xr.open_dataset(DIR+X2_FINAME)[X2VAR2][:N_days_100yrs]
    X2train = xr.concat([X2trainRMM1,X2trainRMM2], dim = 'new_dim') # 2xtime

    # predictand [time]
    Ytrain = xr.open_dataset(DIR+Y_FINAME)[YVAR][:N_days_100yrs]


    # VALIDATION (100 years)
    # repeat for validation data
    X1val = xr.open_dataset(DIR+X1_FINAME)[X1VAR][N_days_100yrs:N_days_100yrs*2]

    X2valRMM1 = xr.open_dataset(DIR+X2_FINAME)[X2VAR1][N_days_100yrs:N_days_100yrs*2]
    X2valRMM2 = xr.open_dataset(DIR+X2_FINAME)[X2VAR2][N_days_100yrs:N_days_100yrs*2]
    X2val = xr.concat([X2valRMM1,X2valRMM2], dim = 'new_dim') # 2xtime

    # predictand [time]
    Yval = xr.open_dataset(DIR+Y_FINAME)[YVAR][N_days_100yrs:N_days_100yrs*2]
    
    
    #adjust dates: 
    print(N_z500runmean)
    print(Yval)
    
    Ytrainroll = Ytrain.rolling(time=N_z500runmean, min_periods=N_z500runmean, center=False).mean('time').dropna('time')
    dates,dates_a2a = days_in_year(Ytrain)
    Ytrainroll['time'] = dates_a2a[:-(N_z500runmean-1)]

    
    Yvalroll = Yval.rolling(time=N_z500runmean, min_periods=N_z500runmean, center=False).mean('time').dropna('time')
    dates,dates_a2a = days_in_year(Yval)
    Yvalroll['time'] = dates_a2a[:-(N_z500runmean-1)]
    
    
    # end X early, so we don't run out of Y data
    X1train_final = X1train[:-(N_z500runmean-1)][:-1*LEAD] 
    X2train_final = X2train[:,:-(N_z500runmean-1)][:,:-1*LEAD] 

    # shift Y to account for lead
    Ytrain_final = Ytrainroll[LEAD:]


    # repeat for validation data
    # end X early, so we don't run out of Y data
    X1val_final = X1val[:-(N_z500runmean-1)][:-1*LEAD] 
    X2val_final = X2val[:,:-(N_z500runmean-1)][:,:-1*LEAD] 

    # shift Y to account for lead
    Yval_final = Yvalroll[LEAD:]
    
    # check that Ytrain and X1/2train are the same size
    # ----- code here -----
    print('...these should all be the same....')
    print(X1train_final.shape)
    print(X2train_final.shape)
    print(Ytrain_final.shape)

    print(X1val_final.shape)
    print(X2val_final.shape)
    print(Yval_final.shape)
    print('... were they?? ....')
    
    # NOTE: we standardize training, validation and testing using the TRAINING mean & std (or median)

    # standardize Xs using Xtrain
    # ----- code here -----
    X1mean = X1train_final.groupby('time.dayofyear').mean()
    X1std = X1train_final.groupby('time.dayofyear').std()
    X2mean = X2train_final.groupby('time.dayofyear').mean()
    X2std = X2train_final.groupby('time.dayofyear').std()

    X1train_norm = (X1train_final.groupby('time.dayofyear')- X1mean).groupby('time.dayofyear')/(X1std)
    X2train_norm = (X2train_final.groupby('time.dayofyear')- X2mean).groupby('time.dayofyear')/(X2std)

    # preprocess Ys by subtracting Ytrain median
    # ----- code here -----
    Ymed = Ytrain_final.quantile(q=.5,dim='time')
    Ytrain_norm = Ytrain_final - Ymed

    # turn Ys into 0s and 1s
    # ----- code here -----
    Ytrain_norm[Ytrain_norm<=0] = 0 
    Ytrain_norm[Ytrain_norm>0] = 1


    # VALIDATION STANDARDIZATION
    # standardize Xs using Xtrain
    X1val_norm = (X1val_final.groupby('time.dayofyear')- X1mean).groupby('time.dayofyear')/(X1std)
    X2val_norm = (X2val_final.groupby('time.dayofyear')- X2mean).groupby('time.dayofyear')/(X2std)

    # preprocess Ys by subtracting Ytrain median
    # ----- code here -----
    Yval_norm = Yval_final - Ymed

    # turn Ys into 0s and 1s
    # ----- code here -----
    Yval_norm[Yval_norm<=0] = 0 
    Yval_norm[Yval_norm>0] = 1
    
    
    # subset predictand (and predictors) to same number of 0s and 1s
    # if you subtract Ytrain median from Ytrain, you will have balanced classes, by definition
    # therefore, you only need to subset validation (& eventually testing) data to have balanced classes
    n_valzero = np.shape(np.where(Yval_norm==0)[0])[0]
    n_valone  = np.shape(np.where(Yval_norm==1)[0])[0]
    i_valzero = np.where(Yval_norm==0)[0]
    i_valone  = np.where(Yval_norm==1)[0]

    # look into the "subset" function and learn how it works
    X1val_norm, Yval_norm, i_valnew = subset(X1val_norm, Yval_norm, n_valzero, n_valone, i_valzero, i_valone)
    X2val_norm = X2val_norm.isel(time = i_valnew,drop=True)
    
    # convert data from xarray to numpy (xarray takes ALOT longer to train with than numpy)
    # ----- code here -----
    X1_train = X1train_norm.T.values
    X2_train = X2train_norm.T.values

    Y_train = Ytrain_norm.values

    X1_val = X1val_norm.T.values
    X2_val = X2val_norm.T.values

    Y_val = Yval_norm.values
    
    ##adding memory to the loop:
    #clunky bad loop: 
    for ee,num in enumerate(reversed(range(N_daysbefore + 1))):
        X1_train_back = X1_train[num:-(ee+1)] 
        X2_train_back = X2_train[num:-(ee+1),:]
        if ee==0:
            X1_train_norm_mem=X1_train[num:-(ee+1)]
            X2_train_norm_mem=X2_train[num:-(ee+1),:]
        else: 
            X1_train_norm_mem = np.vstack([X1_train_back,X1_train_norm_mem])
            X2_train_norm_mem = np.concatenate([X2_train_back,X2_train_norm_mem],axis=1)
    Y_train_mem=Y_train[N_daysbefore:-1] #adjust Ytarget.... 
    X1_train_norm_mem = X1_train_norm_mem.T

    #clunky bad loop: 
    for ee,num in enumerate(reversed(range(N_daysbefore + 1))):
        X1_val_back = X1_val[num:-(ee+1)] 
        X2_val_back = X2_val[num:-(ee+1),:]
        if ee==0:
            X1_val_norm_mem=X1_val[num:-(ee+1)]
            X2_val_norm_mem=X2_val[num:-(ee+1),:]
        else: 
            X1_val_norm_mem = np.vstack([X1_val_back,X1_val_norm_mem])
            X2_val_norm_mem = np.concatenate([X2_val_back,X2_val_norm_mem],axis=1)

    Y_val_mem=Y_val[N_daysbefore:-1] #adjust Ytarget.... 
    X1_val_norm_mem = X1_val_norm_mem.T
    
    print(params)
    
    # variables:
    SEED = params['SEED']
    DROPOUT_RATE = params['DROPOUT_RATE']

    MODELNAME1 = 'ENSO'
    RIDGE1 = params['RIDGE1']
    HIDDENS1 = params['HIDDENS1']
    INPUT_SHAPE1 = np.shape(X1_train_norm_mem)[1:][0]

    MODELNAME2 = 'MJO'
    RIDGE2 = params['RIDGE2']
    HIDDENS2 = params['HIDDENS2']
    INPUT_SHAPE2 = np.shape(X2_train_norm_mem)[1:][0]

    BATCH_SIZE = params['BATCH_SIZE']
    N_EPOCHS = 10000
    PATIENCE = params['PATIENCE'] # number of epochs of no "improvement" before training is stopped
    LR = params['LR'] # learning rate
    
    
    # -------------------------------------------------
    # ENSO MODEL
    model1, input1 = build_model(SEED,
                                 DROPOUT_RATE,
                                 RIDGE1,
                                 HIDDENS1,
                                 INPUT_SHAPE1,
                                 MODELNAME1)


    # MJO MODEL
    model2, input2 = build_model(SEED,
                                 DROPOUT_RATE,
                                 RIDGE2,
                                 HIDDENS2,
                                 INPUT_SHAPE2,
                                 MODELNAME2)   

    # COMBINE ENSO & MJO MODEL
    model = fullmodel(model1, model2,
                      input1, input2,
                      SEED)

    print(model.summary())
    # ------ Training Hyperparameters ------
    optimizer = tf.optimizers.Adam(learning_rate = LR,)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)]                             

    # ------ Compile Model -----
    model.compile(optimizer = optimizer,
                  loss = loss_func,
                  metrics = metrics)

    # ----- Callbacks -----
    ES = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'auto',
                                          patience = PATIENCE, verbose = 0, restore_best_weights = True)
    LR = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=0)


    history = model.fit({MODELNAME1:X1_train_norm_mem,
                         MODELNAME2:X2_train_norm_mem}, 
                        Y_train_mem, 
                        batch_size = BATCH_SIZE, 
                        epochs = N_EPOCHS, 
                        validation_data = ({MODELNAME1:X1_val_norm_mem,
                                            MODELNAME2:X2_val_norm_mem},
                                           Y_val_mem),  
                        verbose = 1,
                        callbacks=[ES,LR],
                        )

    #----- PLOT THE RESULTS -----
    pred = model.predict((X1_val_norm_mem,X2_val_norm_mem))
    cat_pred = np.argmax(pred,axis=1)
    true = Y_val_mem

    acc = accuracy_score(true, cat_pred)
    print('accuracy of network: ', acc)
    
    # Save the weights
    model_dir = params['model_dir']
    print('saving model name: ', model_dir+EXP_NAME+'_'+"{:05}".format(SEED)+'.h5')
    model.save_weights(model_dir+EXP_NAME+'_'+"{:05}".format(SEED)+'.h5')