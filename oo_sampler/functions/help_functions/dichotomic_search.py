# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:09:15 2016

@author: alex
"""
import numpy as np
def f_dichotomic_search(intervall, function, N_max_steps=100, tolerance=0.0000001):
    """
        function that does a dichotomic for the root of a function
    """
    n_iter = 0
    x_inter = intervall.mean()
    x_left = intervall[0]
    x_right = intervall[1]
    f_inter = function(x_inter)
    f_left = function(x_left)
    while n_iter<N_max_steps:
        # if same sign on the left side, go right
        if np.sign(f_left)==np.sign(f_inter):
            x_left = x_inter
            x_inter = (x_left+x_right)/2.
            f_inter = function(x_inter)
            f_left = function(x_left)
        else:
            x_right = x_inter
            x_inter = (x_left+x_right)/2.
            f_inter = function(x_inter)
            f_left = function(x_left)
        n_iter = n_iter + 1
        if np.abs(f_inter)<tolerance:
            #print "break because of tolerance"
            #print "iterations "+str(n_iter)
            return x_inter
    #print "break because of number of iteration"
    return x_inter


if __name__ =="__main__":
    intervall = np.array([-10.,10.])
    def test_function(x):
        return x**3
    print f_dichotomic_search(intervall, test_function)
