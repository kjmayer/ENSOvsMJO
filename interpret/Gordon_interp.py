#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:20:45 2022

@author: Emily Gordon
@editor: Kirsten Mayer
"""
import numpy as np
import tensorflow as tf

def confvacc(confval,predval,Ytest):
    conf_acc = np.zeros(shape=(100))
    conf_percision = np.zeros(shape=(100,2))
    conf_recall = np.zeros(shape=(100,2))

    for per in np.arange(0,100,1):
        conf_thresh = np.percentile(confval,q=per)

        # where prediction confidence is > threshold (location of confident prediction)
        i_conf_predval = np.where(confval > conf_thresh)[0]

        # predicted values (0 or 1) where confidence is > threshold
        conf_predval = predval[i_conf_predval]

        # where confident prediction is correct
        i_corrconf_predval = np.where(Ytest[i_conf_predval]==conf_predval)[0]

        conf_acc[per] = (len(i_corrconf_predval) / len(i_conf_predval)) * 100

        # other metrics:
        confusion = tf.math.confusion_matrix(Ytest[i_conf_predval],conf_predval,num_classes=2)
        tn = confusion[0,0]
        fp = confusion[0,1]
        fn = confusion[1,0]
        tp = confusion[1,1]

        conf_percision[per,0] = (tn/(tn+fn)) * 100
        conf_percision[per,1] = (tp/(tp+fp)) * 100
        conf_recall[per,0]    = (tn/(tn+fp)) * 100
        conf_recall[per,1]    = (tp/(tp+fn)) * 100
    
    return conf_acc, conf_percision, conf_recall

def iconfcorr(confval,predval,Ytest,per=80):
    conf_thresh = np.percentile(confval,q=per)

    # where prediction confidence is > threshold (location of confident prediction)
    i_conf_predval = np.where(confval > conf_thresh)[0]

    # predicted values (0 or 1) where confidence is > threshold
    conf_predval = predval[i_conf_predval]

    # where confident prediction is correct
    i_corrconf_predval = np.where(Ytest[i_conf_predval]==conf_predval)[0]
    
    return i_conf_predval, i_corrconf_predval



def getweights(model,nodestr):
    node = model.get_layer(nodestr) # output node of interest (lower or upper == 0 or 1)
    w1 = np.asarray(node.weights[0][0])[0] # weight from model 1
    w2 = np.asarray(node.weights[0][1])[0] # weight from model 2
    
    return w1,w2

def getoutputs_mapwise(wlower,wupper,a,b):
    
    output_lower = wlower*a
    output_upper = wupper*b
    
    outputvec = np.concatenate((output_lower,output_upper),axis=1)
    
    return outputvec

def getoutputvecs(model,model1,model2,X1test,X2test):
    
    wlower = getweights(model,'lower') # 2 weights (one from each model) for lower class
    wupper = getweights(model,'upper') # 2 weights (one from each model) for upper class
    
    [a,b] = model1.predict(X1test) # prediction from model1 (no softmax)
    [c,d] = model2.predict(X2test) # prediction from model2 (no softmax)
    
    # for each class: model 1 output * weight for model 1 output into final layer
    model1_output_contribution = getoutputs_mapwise(wlower[0],wupper[0],a,b)
    # for each class: model 2 output * weight for model 2 output into final layer
    model2_output_contribution = getoutputs_mapwise(wlower[1],wupper[1],c,d)
    
    return model1_output_contribution, model2_output_contribution
