#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:24:47 2017

@author: francoisdarmon and yousrisellami
"""
import numpy as np

class AbstractStudent:
    """
    Abstract class that defines a student
    """
    def __init__(self,n_a,n_c):
        self.n_a=n_a
        self.n_c=n_c
    
    def exercize(self, activity):
        """
        Function called when an activity with parameters is proposed to the
        student
        """
        assert(activity in np.arange(self.n_a))
        return True
    

class Q_Student(AbstractStudent):
    """
    Model of Q student : student who can do any type of exercize if its
    KC is high enough
    """
    def __init__(self,R_table,initKC,learning_rates,alpha,beta):
        """
        Params : 
            R_table : table of size n_a*n_c KC requirement for each activity
            initKC : Initial KC values
            learning_rates : Rate of learning for each KC
            alpha : KC update additive parameter
            beta : KC update multiplicative parameter
        """
        super().__init__(n_a=R_table.shape[0],n_c=R_table.shape[1])
        self.R_table=R_table
        self.KC=initKC
        self.learning_rates=learning_rates
        self.alpha=alpha
        self.beta=beta
        
    def exercize(self,activity):
        super().exercize(activity)
        
        success_probs=1/(1+np.exp(-self.beta*(self.KC-self.R_table[activity])-self.alpha))
        success_prob=np.prod(success_probs)**(1/success_probs.shape[0])
        print(success_prob)
        success=np.random.uniform()<success_prob
        
        if success:
            self.update_KC(activity)
        
        return success
        
    def update_KC(self,activity):
        """
        Update each KC when the activity was a success 
        """
        print(self.R_table[activity])
        self.KC=self.KC+self.learning_rates*\
                np.maximum(self.R_table[activity]-self.KC,0)
        
        