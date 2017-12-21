#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:09:39 2017

@author: yousri
"""
import numpy as np
from numpy import random
from R_table import R_table




def predefined_sequence(student,R_table_model,T):
    
    
    ## sequence of activities correponding to 10 stages of difficulty
    
    sequence=[[0,0,1,0],[1,0,1,0],[1,2,1,0],[2,0,1,0],[2,2,1,0],
               [3,0,1,1],[3,0,0,1],[3,1,0,1],[4,1,0,1],[5,1,0,1]]
    
    n_c=R_table_model.n_c
    n_p=R_table_model.n_p
    n_a_list=R_table_model.n_a
    
    #### initialization of the student true competences (KC) 
    c_true = np.zeros((n_c,T))
    c_true[:,0]=student.KC
    
    ### stage of difficulty
    stage = 0
    
    activity_list = -1*np.ones((T,n_p))
    answer_list=[[],[],[],[],[],[],[],[],[],[]]

    
    for t in range(T-1):
        
        ## choose exercice corresponding to the current stage of difficulty
        a = np.array(sequence[stage])
        activity_list[t,:]=a
        
        ## return anwser of the student and update its true competence
        answer = student.exercize(a) 
        c_true[:,t+1]=student.KC
            
        answer_list[stage].append(answer)
        
        ### policy to move to the next stage
        ### depends on the previous results of the student
        
        if (stage <= 4):
            if (len(answer_list[stage])>=2 and sum(answer_list[stage][-2:])==2):
                stage+=1
        
        else :
            if (len(answer_list[stage])>=4 and sum(answer_list[stage][-4:])>=3):
                stage+=1
            
        stage = min(stage,9)
            
    

    return activity_list[:-1],c_true,answer_list
        
        