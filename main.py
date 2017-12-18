#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:34:00 2017

@author: yousrisellami and francoisdarmon
"""
import numpy as np 
import student
import riarit
from matplotlib import pyplot as plt
from R_table import R_table
#### Set parameters ######
        
### number of rounds
T = 1000
### number of simulations 
nb_simu = 100

### number of activities 
n_a = 5

### number of competences
n_c = 2

gamma = 0.1

alpha_c_hat = 0.1

#R_table = np.random.uniform(0.8,1,size=[n_a,n_c])
R_table_model = R_table([np.arange(n_a*n_c).reshape(n_a,n_c).astype(float)/(n_a*n_c)])

#initKC = np.clip(np.random.normal(0.4,scale=0.1,size=n_c),a_min=0,a_max=1)
initKC = np.zeros(n_c)

learning_rates = np.random.uniform(low =0.01,high =0.03,size=n_c)


success_prob=0.8 # probability of sucess when KC=R_table
alpha=np.log(success_prob/(1-success_prob))
beta = 7

beta_w = 1 ## coefficient of the previous value w_a
eta_w = 0.2 ## learning rate for w_a
        

student = student.Student(R_table_model,initKC,learning_rates,alpha,beta,lambdas=None)

reward_list,regret_list,activity_list,c_hat,c_true,w_a_history = \
        riarit.Riarit(student,T,R_table_model,beta_w,eta_w,alpha_c_hat,gamma)

#for c in range(n_c):
#    plt.figure()
#    plt.plot(c_true[c,:])
#    plt.plot(c_hat[c,:])
    

#plt.figure()
#for a in range(n_a):
#    plt.plot(w_a_history[0][a,:])
#    
#plt.show()
    

