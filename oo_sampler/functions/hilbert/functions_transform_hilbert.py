# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:18:19 2016
    function that test the hilbert curve on [0,1)^d entries
@author: alex
"""
import numpy as np
from hilbert import Hilbert_to_int
import pdb

def squeeze_to_01(coordinates):
    '''
        function that tranforms the coordinates to [0,1)
        y is a numpy array of dimension p*d
    '''
    return 1/(1+np.exp(-coordinates))

def tranform_to_int_tuple(squeezed_coordinates_single):
    '''
        function that transforms the squeezed coordinates to a tuple of integers
    '''
    d = len(squeezed_coordinates_single)
    big_M = 2**(62/d)
    squeezed_coordinatesM = squeezed_coordinates_single*big_M
    arr =  squeezed_coordinatesM.astype(int) # converts to np.int
    tuple_arr = tuple(arr)
    return [i.item() for i in tuple_arr] # this line is necessary because of the numpy data type

def array_transform_to_hilbert_index(squeezed_coordinates_array):
    '''
        function that applies the hilbert transformation on the array of inputs
    '''
    n, p = squeezed_coordinates_array.shape
    index_array = np.zeros(n)
    for i in range(n):
        int_tuple = tranform_to_int_tuple(squeezed_coordinates_array[i,:])
        index_array[i] = Hilbert_to_int(int_tuple)
    return index_array

def hilbert_sort(coordinates):
    '''
        combine the results
        First squeeze the space to [0,1)
        then obtain the indices corresponding to the points
        then return the order_number of the indices
    '''
    squeezed_coordinates_array = squeeze_to_01(coordinates)
    indices = array_transform_to_hilbert_index(squeezed_coordinates_array)
    permutation = np.argsort(indices)
    return permutation


def resampling_inverse_transform(ordered_u, normalised_weights):
    '''
        following the algorithm in Gerber and Chopin 2015
        return the selected order_numbers
    '''
    N = normalised_weights.shape[0]
    if np.abs(np.sum(normalised_weights)-1.)>0.000001:
        raise ValueError("weights do not sum up to one")
    a_labels = np.zeros(N)
    s = normalised_weights[0]
    m = 0
    for i in range(N):
        while(s<ordered_u[i]):
            m = m + 1
            s = s + normalised_weights[m]
        a_labels[i] = m
    return a_labels

if __name__ == '__main__':
    print "test the functions"
    p = 3
    n = 10
    test = np.random.normal(size=(n,p))
    array_index = array_transform_to_hilbert_index(squeeze_to_01(test))
    print test.shape
    print "array index"
    print array_index
    print "sorted results"
    print hilbert_sort(test)
    if p == 1:
        print "univariate results"
        print np.argsort(test.flatten())

    u = np.random.random(n)
    ordered_u = np.sort(u)
    weights = np.random.random(n)
    normalised_weigths = weights/np.sum(weights)
    normalised_permuted_weights = normalised_weigths[hilbert_sort(test)]
    print ordered_u
    print resampling_inverse_transform(ordered_u, normalised_permuted_weights)
