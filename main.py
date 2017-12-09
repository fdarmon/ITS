#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:34:00 2017

@author: yousrisellami and francoisdarmon
"""
import numpy as np 
from numpy import random
import student
import riarit

#### Set parameters ######
        
### number of rounds
T = 100
### number of simulations 
nb_simu = 100

### number of activities 
n_a = 15

### number of competences
n_c = 5

gamma = 0.1

alpha_c_hat = 0.1

R_table = np.random.uniform(0.8,1,size=[n_a,n_c])

initKC = np.random.normal(0.4,scale=1,size=n_c)

learning_rates = np.random.uniform(0.2,0.1,size=n_c)

alpha=0.5
beta = 0.5

beta_w = 1 ## coefficient of the previous value w_a
eta_w = 0.1 ## learning rate for w_a
        

student = student.Student(R_table,initKC,learning_rates,alpha,beta,lambdas=None)

reward_list,activity_list,c_hat,c_true = riarit.Riarit(student,T,R_table,beta_w,eta_w,alpha_c_hat,gamma)