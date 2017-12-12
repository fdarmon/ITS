# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy import random
from R_table import R_table



def choose_activity(w_a_list,gamma):
    
    """Parameters :
     w_a_list : list of n_p vectors recent rewards provided by each activity
     gamma : exploration parameter
    
        Returns :
     a : vector of activity chosen 
    """
    res=[]
    for w_a in w_a_list:
        n_a = np.size(w_a)
        w_a = (1./np.sum(w_a))*w_a ## normalize the weights 
        
        ### gamma-greedy exploration policy i.e explore with proba gamma and 
        ### exploit with proba 1-gamma.
        
        if (np.random.binomial(1,gamma)==1):
            res.append(np.random.choice(n_a))
        
        else :
            res.append(np.random.choice(n_a, p = w_a))
    
        return(np.array(res))
    
def compute_reward(a,answer, R_table_model,c_hat):
    
    """ Parameters :
    # activity vector a which was proposed
    # answer given by the student
    # R_table_model
    # c_hat : most recent estimated competences
    
       Returns :
    reward : array of size(1 x n_c) reward associated to each competence
    for the activity a, i.e estimated progress on each competence provided
    by activity a.
    """
    n_c = R_table_model.n_c
    reward = np.zeros(n_c)
    if (answer):
        reward = np.maximum(R_table_model.get_KCVector(a)-c_hat,0)
    
    else :
        reward = np.minimum(R_table_model.get_KCVector(a)-c_hat,0)
    
    return reward
        

    
 
    
def Riarit(student,T,R_table_model,beta_w,eta_w,alpha_c_hat,gamma):
    
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
    

    
    ### weights tracking how much rewards each activity is providing
    
    n_c=R_table_model.n_c
    n_p=R_table_model.n_p
    n_a_list=R_table_model.n_a
    
    reward_list = -1*np.ones(T)
    activity_list = -1*np.ones((T,n_p))
    
    
    w_a=[] # list of n_p vector of size (n_a_list[i])
    for n_a in n_a_list:
        w_a.append(0.1*np.ones((n_a))) ### initialization with constant weights
    
    w_a_history=[w_a]
    #### initialization of the student true competences (KC) 
    c_true = np.zeros((n_c,T))
    c_true[:,0]=student.KC
    
    ### initialization of the estimated competence 
    c_hat= np.zeros((n_c,T))
    c_hat[:,0] = 0.1*np.ones(n_c)
    
    

    
    for t in range(T-1):
        
        a = choose_activity(w_a,gamma)
        activity_list[t,:]=a
        
        ## return anwser of the student and update its true competence
        answer = student.exercize(a) 
        c_true[:,t+1]=student.KC
    
        r = compute_reward(a,answer,R_table_model,c_hat[:,t])
        
        ## use the computed rewards to update the estimation of competence
        c_hat[:,t+1] = c_hat[:,t] + alpha_c_hat*r
        
        
        ## update the weights w
        new_w_a=w_a

        for i in range(n_p):
            new_w_a[i][a[i]] = np.clip(beta_w*w_a[i][a[i]] +eta_w*np.sum(r),0,1)
        w_a=new_w_a
        w_a_history.append(w_a)
        
        reward_list[t]= np.sum(r)
        
    
    return reward_list[:-1],activity_list[:-1],c_hat,c_true,w_a_history
        
        
        


        
    
    
    
    















