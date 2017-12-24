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
import baselines

ex_type=np.array([[0.7,0.4,0,0,0,0.5],
    [0.7,0.6,0.3,0,0,0.5],
    [0.7,0.7,0.6,0,0,0.5],
    [1,0.7,0.6,0.5,0.3,0.7],
    [1,0.9,0.7,0.7,0.5,0.7],
    [1,1,1,1,1,1]])

price_presentation=np.array([[0.8,1,1,1,1,0.2],
    [1,1,1,1,1,0.6],
    [0.9,1,1,1,1,1]])

cents_notation=np.array([[0.8,1,1,1,1,1],
    [0.9,1,1,1,1,1]])

money_type=np.array([[1,1,1,0.9,0.9,1],
    [0.1,1,1,1,1,1]]) # 1 for not relevent value (will be ignored in the R_table)



#### Set parameters ######
        
T = 150 # number of rounds
n_c = 6 # KnowMoney IntSum IntDec DecSum DecDec Memory 

gamma = 0.1

alpha_c_hat = 0.1

R_table_model=R_table([ex_type,price_presentation,cents_notation,money_type])
initKC = np.zeros(n_c)
n_p=R_table_model.n_p
learning_rates = np.random.uniform(low =0.1,high =0.15,size=n_c)


success_prob=0.8 # probability of sucess when KC=R_table
alpha=np.log(success_prob/(1-success_prob))
beta = 7

beta_w = 1 ## coefficient of the previous value w_a
eta_w = 0.2 ## learning rate for w_a
 
# %% Evaluation of w_a_history
def gen_w_a_iter(current_student,method):
    if method == "Random":
        reward_list,_,activity_list,_,c_true,w_a_history = \
            riarit.Exp3(current_student,T,R_table_model,beta_w,eta_w,alpha_c_hat,1)

    elif method == 'Riarit':
        reward_list,_,activity_list,_,c_true,w_a_history = \
            riarit.Exp3(current_student,T,R_table_model,beta_w,eta_w,alpha_c_hat,gamma)

    else:
        return
    
    return w_a_history

to_test=["Random","Riarit"]
w_a_mean={method : None for method in to_test}
n_itr=50
T=500
for i in range(n_itr):
    for method in to_test:
        current_student=student.Student(R_table_model,initKC,learning_rates,alpha,beta,lambdas=None)
        w_a_history=gen_w_a_iter(current_student,method)
        if w_a_mean[method] is None:
            w_a_mean[method]=w_a_history
        for j in range(n_p):
            w_a_mean[method][j]=w_a_mean[method][j]+w_a_history[j]/n_itr

# %% Plot the average interest for parameter 0 : exercize type
plt.figure()
plt.plot(w_a_mean["Random"][0])
plt.legend(["{}".format(i) for i in range(R_table_model.n_a[0])])
            
# %% Plot the evolution of competences  for one student 
def gen_progresses_iter(current_student,method):
    """
    Generate one iteration of giving T activity to one student with method
    returns the KC of the student function of time

    """
    if method == "Predefined sequence":
        activity_list,c_true,_ = \
            baselines.predefined_sequence(current_student,R_table_model,T)

    elif method == "Random":
        reward_list,_,activity_list,_,c_true,_ = \
            riarit.Exp3(current_student,T,R_table_model,beta_w,eta_w,alpha_c_hat,1)

    elif method == 'Riarit':
        reward_list,_,activity_list,_,c_true,_ = \
            riarit.Exp3(current_student,T,R_table_model,beta_w,eta_w,alpha_c_hat,gamma)

    else:
        return
    
    return c_true
      
n_itr=50
methods=["Predefined sequence","Random","Riarit"]
true_KC = {method : None for method in methods}
activities = {method : None for method in methods}
for i in range(n_itr):
    for method in methods:
        current_student=student.Student(R_table_model,initKC,learning_rates,alpha,beta,lambdas=None)
        KC_iter=gen_progresses_iter(current_student,method)
        if true_KC[method] is None:
            true_KC[method]=KC_iter/n_itr
        else:
            true_KC[method]=true_KC[method]+KC_iter/n_itr

for c in range(n_c):
    plt.figure()
    for method in methods:
        plt.plot(true_KC[method][c],label=method)
        
    plt.legend()
    plt.title("Evolution of competence level {} for several methods".format(c))

# %%
"""
cumulative_reward=np.cumsum(reward_list_3)
plt.figure()
plt.plot(regret_list_3+cumulative_reward,label="Optimal")

plt.plot(np.cumsum(reward_list_3),label='Exp3')
plt.title("Cumulative reward")
plt.legend()
"""   


