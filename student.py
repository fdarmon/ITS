#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:24:47 2017

@author: francoisdarmon
"""

class AbstractStudent:
    """
    Abstract class that defines a student
    """
    
    def exercize(self, activity, parameters):
        """
        Function called when an activity with parameters is proposed to the
        student
        """
        reward=0
        return reward