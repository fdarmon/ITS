# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy import random
from R_table import R_table



def ZPD(c_hat):
     
    """Parameters :
    c_hat : list of estimated competence of size n_C
    
        Returns :
     zpd : vector of size n_p, each element represents the maximal index 
     of parameter value that can be chosen to form an activity depending
     on the estimated competences of the student
     """
    
    
    ### maximum level of difficulty that can be proposed 
    difficulty=1
    
    if (c_hat[1]>0.25):
        difficulty=2
        
        if (c_hat[1]>0.4):
            difficulty=3
            
            if (c_hat[2]>0.4):
                difficulty=4
                
                if (c_hat[3]>0.3):
                    difficulty=5
                
                    if (c_hat[3]>0.5):
                        difficulty=6
    
    ### maximum level of presentation that can be proposed
    ## level 0: spoken and written, level 1: written, level 2 :spoken 
    presentation=1
    
    if (c_hat[-1]>0.05):
        presentation=2
        
        if (c_hat[-1]>0.1):
            presentation=3
             
    zpd = np.array([difficulty,presentation,2,2])
             
    return zpd
            
    
    

def choose_activity(w_a_list,gamma,zpd):
    
    """Parameters :
     w_a_list : list of n_p vectors recent rewards provided by each activity
     gamma : exploration parameter
    zpd : array of indices that set the eligible parameters values
        
        Returns :
     a : vector of activity chosen size n_p
    """
    res=[]
    i=0
    for w_a in w_a_list:
        
        w_a = w_a[:zpd[i]]
        n_a = np.size(w_a)
        w_a = (1./np.sum(w_a))*w_a ## normalize the weights 
        i+=1
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
    best_activity_list = -1*np.ones((T,n_p))
    regret_list=-1*np.ones(T)
    
    w_a=[] # list of n_p vector of size (n_a_list[i])
    for n_a in n_a_list:
        w_a.append(0.2*np.ones((n_a))) ### initialization with constant weights
       # w_a.append(np.random.uniform(0.1,0.3,n_a))
        
    w_a_history=[w_a]
    #### initialization of the student true competences (KC) 
    c_true = np.zeros((n_c,T))
    c_true[:,0]=student.KC
    
    ### initialization of the estimated competence 
    c_hat= np.zeros((n_c,T))
    c_hat[:,0] = 0.1*np.ones(n_c)
    
    zpd = np.array([1,1,2,2])

    
    for t in range(T-1):
        
        #print w_a_history
        a = choose_activity(w_a,gamma,zpd)
        activity_list[t,:]=a
        best_activity_list[t,:]=student.get_best_activity()[0]
        
        ## return anwser of the student and update its true competence
        answer = student.exercize(a) 
        c_true[:,t+1]=student.KC
        
        r = compute_reward(a,answer,R_table_model,c_hat[:,t])
        
        ## use the computed rewards to update the estimation of competence
        c_hat[:,t+1] = c_hat[:,t] + alpha_c_hat*r
        
        # update zpd
        zpd = ZPD(c_hat[:,t+1])
        
        ## update the weights w

        for i in range(n_p):
            w_a[i][a[i]] = np.clip(beta_w*w_a[i][a[i]] +eta_w*np.sum(r),0,1)
            #w_a[i][a[i]]=max(0,w_a[i][a[i]])
        w_a_history.append(w_a)
        
        
        
        rew=np.sum(r)
        reward_list[t+1]= rew
        regret_list[t+1]=regret_list[t]+student.get_best_activity()[1]-rew

    
    return reward_list[:-1],regret_list[:-1],activity_list[:-1],c_hat,c_true,\
w_a_history,best_activity_list[:-1]
        
        



        
    
#c_hat=np.array([0.5,0.5,0.3,0.4,0.1,0.07])
#print ZPD(c_hat)


############### EXP3 ########################################################


def choose_activity_exp3(w_a_list,gamma,zpd):
    
    """Parameters :
     w_a_list : list of n_p vectors recent rewards provided by each activity
     gamma : exploration parameter
    zpd : array of indices that set the eligible parameters values
        
        Returns :
     a : vector of activity chosen size n_p
     winner_prob : array of shape n_p with probabilities associated
     to each winner arm.
    """
    res=[]
    winner_prob=[]
    i=0
    for w_a in w_a_list:
        
        w_a = w_a[:zpd[i]] ## only select parameters eligible 
        n_a = np.size(w_a)
        w_a = (1./np.sum(w_a))*w_a ## normalize the weights 
        i+=1
        
        ### compute probabilities used to draw arms
        probabilities = (1-gamma)*w_a + float(gamma)/n_a
         
        ## select  an arm = parameter value 
        arm = np.random.choice(n_a,p=probabilities)
        res.append(arm)
        winner_prob.append(probabilities[arm])
    
    return(np.array(res)),np.array(winner_prob)


def Exp3(student,T,R_table_model,beta_w,eta_w,alpha_c_hat,gamma):
    
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
    regret_list=-1*np.ones(T)
    
    w_a=[] # list of n_p vector of size (n_a_list[i])
    for n_a in n_a_list:
        w_a.append(np.ones((n_a))) ### initialization with constant weights
      
        
    w_a_history=[w_a]
    #### initialization of the student true competences (KC) 
    c_true = np.zeros((n_c,T))
    c_true[:,0]=student.KC
    
    ### initialization of the estimated competence 
    c_hat= np.zeros((n_c,T))
    c_hat[:,0] = 0.1*np.ones(n_c)
    
    zpd = np.array([1,1,2,2])

    
    for t in range(T-1):
        
        a,winner_prob = choose_activity_exp3(w_a,gamma,zpd)
        activity_list[t,:]=a
        
        ## return anwser of the student and update its true competence
        answer = student.exercize(a) 
        c_true[:,t+1]=student.KC
        
        r = compute_reward(a,answer,R_table_model,c_hat[:,t])
        
        ## use the computed rewards to update the estimation of competence
        c_hat[:,t+1] = c_hat[:,t] + alpha_c_hat*r
        
        # update zpd
        zpd = ZPD(c_hat[:,t+1])
        
        ## update rewards
        
        rew=np.sum(r)
        reward_list[t+1]= rew
        
        ## update the weights w

        for i in range(n_p):
            
            ## use importance weight trick to estimate rewards
            r_hat = rew/winner_prob[i]

            w_a[i][a[i]] = w_a[i][a[i]]*np.exp((gamma*r_hat)/n_a_list[i])
           
        w_a_history.append(w_a)
        
        
        
        

    
    return reward_list[:-1],activity_list[:-1],c_hat,c_true,w_a_history









