#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
@author: Emily Gordon
@editor: Kirsten Mayer
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Verdana']
plt.rcParams.update({'font.size': 15})
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([])
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
dpiFig = 300.



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


# functions for generating model from Gordon et al. (in prep)

# ---------------- build individual model ----------------
def build_model(seed,dropout_rate,activity_reg,hiddens,input_shape,name, biasbool=True):
    
    dense_layers = len(hiddens)
    
    inputs = tf.keras.Input(shape=(input_shape,),name=name)
    x = layers.Dense(hiddens[0], activity_regularizer=regularizers.l2(activity_reg),
                           bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                           kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                           activation='relu',
                           use_bias=biasbool)(inputs)
    
    for i in range(dense_layers-1):
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(hiddens[i+1],
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activation='relu',
                               use_bias=biasbool)(x) ## dense layer   
    
    a = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg),
                               use_bias=biasbool)(x)
    b = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg),
                               use_bias=biasbool)(x)
    
    model = tf.keras.Model(inputs=inputs,outputs=[a,b])
    
    return model,inputs

# ---------------- linearly combine output from one class (from each model)----------------
def singlenodemodel(seed,singlenodename):
    
    inputlayer1 = layers.Input(1)
    inputlayer2 = layers.Input(1)
    concatlayer = layers.Concatenate()([inputlayer1,inputlayer2])
    
    dense = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               use_bias=False,
                               )(concatlayer)
    model = tf.keras.Model(inputs=[inputlayer1,inputlayer2],outputs=dense,name=singlenodename)
    
    return model

# ---------------- concatenate combined output & apply softmax ----------------
def finallayer():
    
    inputlayer1 = layers.Input(1,)
    inputlayer2 = layers.Input(1,)

    concatlayer = layers.Concatenate()([inputlayer1,inputlayer2])
    final = tf.keras.layers.Softmax()(concatlayer)
    lastmodel = tf.keras.Model(inputs=[inputlayer1, inputlayer2], outputs = final)
    
    return lastmodel

# ---------------- construct full model (build_model, singlenodemodel, finallayer) ----------------
def fullmodel(model1,model2,input1,input2,seed):

    a,b = model1(input1) # define the outputs of individual NNs
    c,d = model2(input2)
    
    singlenodemodel1 = singlenodemodel(seed,'lower') #define a model that takes two inputs to a single node with linear activation
    singlenodemodel2 = singlenodemodel(seed,'upper')
    
    singlenodea = singlenodemodel1([a, c]) #pass the individual NN outputs in pairs to the corresponding node
    singlenodeb = singlenodemodel2([b, d]) 
    
    lastmodel = finallayer() # define a model that takes nodes, concats and adds a softmax    
    finaloutput = lastmodel([singlenodea, singlenodeb]) # point single nodes to the model
    
    model = tf.keras.Model(inputs = [input1, input2], # and finally point the initial first layer 
                           outputs = [finaloutput])   # to the final final layer!
    
    return model

# ---------------- Learning Rate Callback Function ----------------
def scheduler(epoch, lr):
    # This function keeps the initial learning rate for the first ten epochs
    # and decreases it exponentially after that.
    if epoch < 10:
        return lr
    else:
        return lr * tf.constant(.1,dtype=tf.float32)


# %% ---------------- Plot Accuracy & Loss during Training ----------------
def plot_results(history, exp_info, showplot=True):
    
    n_epochs, patience = exp_info
    
    trainColor = 'k'
    valColor = (141/255,171/255,127/255,1.)
    FS = 14
    plt.figure(figsize=(15, 7))
    
    #---------- plot loss -------------------
    ax = plt.subplot(2,2,1)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

    plt.plot(history.history['sparse_categorical_accuracy'], 'o', color=trainColor, label='Training',alpha=0.6)
    plt.plot(history.history['val_sparse_categorical_accuracy'], 'o', color=valColor, label='Validation',alpha=0.6)
    plt.vlines(len(history.history['val_sparse_categorical_accuracy'])-(patience+1),-10,np.max(history.history['loss']),'k',linestyle='dashed',alpha=0.4)

    plt.title('ACCURACY')
    plt.xlabel('EPOCH')
    plt.xticks(np.arange(0,n_epochs+20,20),labels=np.arange(0,n_epochs+20,20))
    plt.yticks(np.arange(.4,1.1,.1),labels=[0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # 0,0.1,0.2,0.3,
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FS)
    plt.xlim(-2, n_epochs)
    plt.ylim(.4,1)
    
    # ---------- plot accuracy -------------------
    ax = plt.subplot(2,2,2)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
    
    plt.plot(history.history['loss'], 'o', color=trainColor, label='Training',alpha=0.6)
    plt.plot(history.history['val_loss'], 'o', color=valColor, label='Validation',alpha=0.6)
    plt.vlines(len(history.history['val_loss'])-(patience+1),0,1,'k',linestyle='dashed',alpha=0.4)
    plt.title('PREDICTION LOSS')
    plt.xlabel('EPOCH')
    plt.legend(frameon=True, fontsize=FS)
    plt.xticks(np.arange(0,n_epochs+20,20),labels=np.arange(0,n_epochs+20,20))
    plt.yticks(np.arange(0,1.1,.1),labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.ylim(0,1)
    plt.grid(True)
    plt.xlim(-2, n_epochs)

    # ---------- Make the plot -------------------
    #plt.tight_layout()
    if showplot==False:
        plt.close('all')
    else:
        plt.show()






