#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:10:05 2017

@author: francoisdarmon
"""
import numpy as np 
import student
import riarit
from matplotlib import pyplot as plt
from R_table import R_table

ex_type=np.array([[0.7,0.4,0,0,0,0.5],
    [0.7,0.6,0.3,0,0,0.5],
    [0.7,0.7,0.6,0,0,0.5],
    [1,0.7,0.6,0.5,0.3,0.7],
    [1,0.9,0.7,0.7,0.5,0.7],
    [1,1,1,1,1,1]])

price_presentation=np.array([[0.9,1,1,1,1,1],
    [1,1,1,1,1,0.6],
    [0.8,1,1,1,1,0.2]])

cents_notation=np.array([[0.8,1,1,1,1,1],
    [0.9,1,1,1,1,1]])

money_type=np.array([[1,1,1,0.9,0.9,1],
    [0.1,-1,-1,1,1,1]]) # 1 for not relevent value (will be ignored in the R_table)



#### Set parameters ######
        
T = 1000 # number of rounds
nb_simu = 100 # number of simulations
n_c = 6 # KnowMoney IntSum IntDec DecSum DecDec Memory 

gamma = 0.1

alpha_c_hat = 0.1

R_table_model=R_table([ex_type,price_presentation,cents_notation,money_type])
initKC = np.zeros(n_c)

learning_rates = np.random.uniform(low =0.05,high =0.1,size=n_c)


success_prob=0.8 # probability of sucess when KC=R_table
alpha=np.log(success_prob/(1-success_prob))
beta = 7

beta_w = 1 ## coefficient of the previous value w_a
eta_w = 0.2 ## learning rate for w_a
        

student = student.Student(R_table_model,initKC,learning_rates,alpha,beta,lambdas=None)

reward_list,activity_list,c_hat,c_true,w_a_history = \
        riarit.Riarit(student,T,R_table_model,beta_w,eta_w,alpha_c_hat,gamma)

for c in range(n_c):
    plt.figure()
    plt.plot(c_true[c,:])
    plt.plot(c_hat[c,:])
    
plt.figure()

    
    
    


