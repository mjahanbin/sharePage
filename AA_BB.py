# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:29:15 2025

@author: w10
"""
from  functools import cache

@cache
def getan_rec(k, n):
    if n < 0:
        a_n = 0
    elif n == 0:
        a_n = k
    else:
        a_n_1 = getan_rec(k, n-1)
        a_n_2 = getan_rec(k, n-2)
        a_n = k*k * a_n_1 - a_n_2

    return a_n


def ABfromK(k, n):
    a_n = getan_rec(k, n)
    a_n_1 = getan_rec(k, n-1)
    
    A = a_n
    B = a_n * k*k - a_n_1
    
    return A,B


if __name__ == '__main__':
    K = 29438
    for n in range(10):
        A,B = ABfromK(K, n)
        print (A,B,K)
    
    
        
    