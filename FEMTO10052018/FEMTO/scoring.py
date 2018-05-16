#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:50:21 2018

@author: ubuntusamuel
"""
import numpy as np
import math 

def score(delta_rul):
    if delta_rul < 0:
        return (math.e**(math.log(0.5)*(-delta_rul)/5))
    elif delta_rul >= 0:
        return (math.e**(math.log(0.5)*(delta_rul)/20))
    
def scoring_fun(rul, p_rul):
    scr = []
    for i in range(len(rul)):
        s = 100*(p_rul[i]-rul[i])/(1.*rul[i])
        scr.append(score(s))
    return np.mean(np.array(scr))

def rmse(rul, p_rul):
    scr = []
    
    for i in range(len(rul)):
        s = p_rul[i]-rul[i]
        scr.append(s**2)
    return np.sqrt(np.mean(np.array(scr)))