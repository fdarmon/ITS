#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:24:47 2017

@author: francoisdarmon and yousrisellami
"""
import numpy as np

class Student():
    """
    Model of P and Q students (depending on lambdas parameters)
    """
    def __init__(self,R_table,initKC,learning_rates,alpha,beta,lambdas=None):
        """
        Params : 
            R_table : table of size n_a*n_c KC requirement for each activity
            initKC : Initial KC values
            learning_rates : Rate of learning for each KC
            alpha : KC update additive parameter
            beta : KC update multiplicative parameter
            lambdas : lambda vector (constant to 1 if Q_student)
        """
        self.R_table=R_table
        self.KC=initKC
        self.learning_rates=learning_rates
        self.alpha=alpha
        self.beta=beta
        self.n_a=R_table.shape[0]
        self.n_c=R_table.shape[1]
        
        # Define lambdas : size n_a, ponderation close to 0 for P student
        if lambdas is None:
            self.lambdas=np.ones((self.n_a,))
        else:
            self.lambdas=lambdas
        
    def exercize(self,activity):        
        success_probs=self.lambdas[activity]/(1+np.exp(-self.beta*(self.KC-self.R_table[activity])-self.alpha))
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
        
        