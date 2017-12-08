# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy import random




def choose_activity(w_a,gamma):
    
    """Parameters :
     w_a : recent rewards provided by each activity
     gamma : exploration parameter
    
        Returns :
     a : index of activity chosen 
    """
    w_a = (1./np.sum(w_a))*w_a ## normalize the weights 
    
    ### gamma-greedy exploration policy i.e explore with proba gamma and 
    ### exploit with proba 1-gamma.
    
    if (np.random.binomial(1,gamma)==1):
        return np.random.choice(n_a)
    
    else :
        return np.random.choice(n_a, p = w_a)
    
    
def compute_reward(a,answer, R_table,c_hat):
    
    """ Parameters :
    # activity a which was proposed
    # answer given by the student
    # R_table, c_hat
    
       Returns :
    reward : array of size(1 x n_c) reward associated to each competence
    for the activity a, i.e estimated progress on each competence provided
    by activity a.
    """
    reward = np.zeros(n_c)
    
    if (answer):
        
        reward = np.array([max(R_table[a,i]-c_hat[i,t],0) for i in range(n_c)])
    
    else :
         reward = np.array([max(c_hat[i,t]-R_table[a,i],0) for i in range(n_c)])
    
    return reward
        

    
    
def Riarit(T,R_table,beta_w,eta_w,alpha_c_hat,gamma):
    
     """Parameters:
    
    T :number of rounds : at each round, an activity is proposed to each students
    R_table : array of shape (n_a x n_c) with elements qi(a) representing the
   competence levels required in KC i to have maximal success in activity a.
    beta_w, eta_w : parameters used to update the estimations w_a
    alpha_c_hat : learning rate for c_hat the estimated competence
    gamma : exploration parameter 
    
        Returns :
            
    reward_list : reward  received at each round
    activity_list : activity chosen at each round
    c_hat : estimated level for each competence at each round
    
    """
    
    
    reward_list = -1*np.ones(T)
    
    activity_list = -1*np.ones(T)
    
    ### weights tracking how much rewards each activity is providing
    
    w_a = 0.1*np.ones(n_a) ### initialization with uniform weights
    
    ### initialization of the estimated competence 
    c_hat = 0.1* np.ones((n_c,T))
    
    
    
    
    for t in range(T):
        
        a = choose_activity(w_a,gamma)
        activity_list[t]=a
        
        ## return anwser of the student and update its true competence
        answer = student.exercize(self,a) 
    
        r = compute_reward(a,answer,R_table,c_hat)
        
        ## use the computed rewards to update the estimation of competence
        c_hat = c_hat + alpha_c_hat*r
        c_hat[t]=c_hat
        
        ## update the weights w
        w_a = beta_w*w_a +eta_w*np.sum(r)
        
        reward_list[t]= np.sum(r)
        
    
    return reward_list,activity_list,c_hat
        
        
        

#### Set parameters ######
        
T = 20

### number of simulations 
nb_simu = 100

### number of activities 
n_a = 10

### number of competences
n_c = 5

gamma = 0.1

alpha_c_hat = 0.1

R_table =  np.zeros((n_a,n_c))
sa
beta_w = 1 ## coefficient of the previous value w_a
eta_w = 0.1 ## learning rate for w_a
        
        
    
    
    
    















