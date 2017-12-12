#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:24:47 2017

@author: francoisdarmon and yousrisellami
"""
import numpy as np
from R_table import R_table

class Student():
    """
    Model of P and Q students (depending on lambdas parameters)
    """
    def __init__(self,R_table_model,initKC,learning_rates,alpha,beta,lambdas=None):
        """
        Params : 
            R_table_model : 
                KC requirement for each activity (R_table object)

            initKC : 
                Initial KC values
            learning_rates : 
                Rate of learning for each KC
            alpha : 
                KC update additive parameter
            beta : 
                KC update multiplicative parameter
            lambdas : 
                lambda vector (constant to 1 if Q_student)
        """
        self.R_table_model=R_table_model
        self.KC=initKC
        self.learning_rates=learning_rates
        self.alpha=alpha
        self.beta=beta
        self.n_a_list=R_table_model.n_a
        self.n_c=R_table_model.n_c
        self.n_p=R_table_model.n_p
        

        
    def exercize(self,activity):    
        """
        Activity must be a numpy array of shape (n_p,)
        """
        assert(activity.shape==(self.n_p,))
        q=self.R_table_model.get_KCVector(activity)
        success_probs=self.get_lambdas(activity)/(1+np.exp(-self.beta*(self.KC-q)-self.alpha))
        success_prob=np.prod(success_probs)**(1./success_probs.shape[0])
        #print(success_prob)
        success=np.random.uniform()<success_prob
        
        if success:
            self.update_KC(q)
        
        return success
        
    def update_KC(self,q):
        """
        Update each KC when the activity was a success 
        q of shape(n_p,) requirement of the activity for each KC
        """
        self.KC=self.KC+self.learning_rates*\
                np.maximum(q-self.KC,0)
        
    def get_lambdas(self,activity):
        # TODO
        return 1