# -*- coding: utf-8 -*-


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
        #print R_table_model.get_KCVector(a)
    else :
        reward = np.minimum(R_table_model.get_KCVector(a)-c_hat,0)
    
    return reward
        

    
 
    
    
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


def Exp3(student,T,R_table_model,alpha_c_hat,gamma,compute_regret=False):
    
    """Parameters:
    
    T :number of rounds : at each round, an activity is proposed to each students
    R_table : array of shape (n_a x n_c) with elements qi(a) representing the
   competence levels required in KC i to have maximal success in activity a.
    alpha_c_hat : learning rate for c_hat the estimated competence
    gamma : exploration parameter 
    
    Returns :
            
    reward_list : reward  received at each round
    regret_list : cumulative regret up to each round
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
      
        
    w_a_history=[np.zeros((T,n_a)) for n_a in n_a_list]
    #### initialization of the student true competences (KC) 
    c_true = np.zeros((n_c,T))
    c_true[:,0]=student.KC
    
    ### initialization of the estimated competence 
    c_hat= np.zeros((n_c,T))
    c_hat[:,0] = 0.5*np.ones(n_c)
    #c_hat[:,0] = np.random.uniform(low=0,high=0.5,size=n_c)
    #c_hat[:,0] = np.zeros(n_c)
    zpd = np.array([1,1,2,2])

    best_activity_list=[]
    correct_answers = 0
    
    for t in range(T):
        
        a,winner_prob = choose_activity_exp3(w_a,gamma,zpd)
        activity_list[t,:]=a
        
        ## return anwser of the student and update its true competence
        answer = student.exercize(a) 
        c_true[:,t]=student.KC
        correct_answers += 1*(answer)
        
        r = compute_reward(a,answer,R_table_model,c_hat[:,t])
        ## update rewards
        
        rew=np.sum(r)
        reward_list[t]= rew
        if compute_regret:
            best_activity,best_reward=student.get_best_activity()
            best_activity_list.append(best_activity)
        else:
            best_activity=0
            best_reward=0
        if (t==0):
            regret_list[0]=best_reward-rew
           
        else:
            regret_list[t]=regret_list[t-1]+best_reward-rew
        
        if (t<=T-2):
            c_hat[:,t+1] = c_hat[:,t] + alpha_c_hat*r
            
        # update zpd
        zpd = ZPD(c_hat[:,t])
        

        

        ## update the weights w
    

        for i in range(n_p):
            
            ## use importance weight trick to estimate rewards
            r_hat = rew/winner_prob[i]
            
            w_a[i][a[i]] = w_a[i][a[i]]*np.exp(gamma*r_hat)
            w_a[i] = (1./np.sum(w_a[i]))*w_a[i] ## normalize the weights to avoid explosion 
            
            w_a[i] = np.maximum(w_a[i],10**(-5)*np.ones(n_a_list[i]))
            w_a[i] = (1./np.sum(w_a[i]))*w_a[i]
            w_a_history[i][t,:]=w_a[i]
            
        
        
        

    
    return reward_list,regret_list,activity_list,c_hat,c_true,w_a_history,best_activity_list,correct_answers
