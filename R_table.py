#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:06:12 2017

@author: francoisdarmon
"""

import numpy as np

class R_table():
    """
    R_table model, KC requirement for each activity
        - Factorized : when each activity is parametrized by n_p parameters
        - Not factorized : the value can directly be read in a table n_a*n_c
    """
    
    def __init__(self,list_tables):
        
        self.list_tables=list_tables
        self.n_p=len(list_tables)
        self.n_c=list_tables[0].shape[1]
        self.n_a=[table.shape[0] for table in list_tables] # list of possible values for each parameters
            
    def get_KCVector(self,activity):
        assert(activity.shape==(self.n_p,))
        p=np.ones(self.n_c)
        for i in range(self.n_p):
            p=p*self.list_tables[i][activity[i],:]
            
        return(p)