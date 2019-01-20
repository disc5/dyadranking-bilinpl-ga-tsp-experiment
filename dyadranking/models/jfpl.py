# -*- coding: utf-8 -*-
"""
Joint-Feature Plackett-Luce Model - Quasi-Newton Impl.

Created on Mon Feb 27 11:53:24 2017
@author: ds
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.optimize import minimize

class JointFeaturePLModel:
    """ 
        Joint-Feature Plackett-Luce model implementation for dyad ranking.
    """
    
    def __init__(self):
        self.model = None
    
    def train(self, jf_ordered_tensor):
        """
            Train a Joint-Feature Plackett-Luce model on sets of rank-ordered joint-feature vectors.
            Note: Instead of "sets" the data is expected to be in the form of tensors.
            
            Args:
            
        """
        def f(x):
            return self.getloss_JF_ordered_tensor(x, jf_ordered_tensor)
        jfvec_dim = jf_ordered_tensor.shape[2]
        weight0 = np.zeros(jfvec_dim, dtype = np.float64)
        result = minimize(f, weight0, method='L-BFGS-B')
        self.model = result.x
    
    def predict(self, example):
        """
            Predict a ranking given a set of joint-feature vectors.
            
            Args:
                example : a set of joint-feature vectors, Nxd matrix (N: number of vectors, d: number of joint-features)
                
            Returns:
                ndarray : an array of ints representing the ordering of indices - this can be used to rank the joint-feature vectors
        """
        scores = np.dot(self.model, np.transpose(example))
        return np.argsort(scores)[::-1] #, scores
    
    def getloss_JF_ordered_tensor(self, weight, jf_ordered_tensor):
        """
            Calculate the negative log likelihood (NLL) function for the JFPL model on ordered (joint feature) tensors.

            The tensors are expected to be ordered along the second dimension.
            First dim: Instances (contexts / people)
            Second dim: Items (labels)
            Third dim: Joint-features (combined features of instances and labels) 
            
            Note: In the case of the Bilinpl model, the joint-feature vectors consist of Kronecker-products
            
            Args:
                weight    : ndarray (1,5)
                jf_ordered_tensor  : ndarray (N x M x (p+1)(q+1))
            
            Returns:
                NLL : scalar value
        """
        (N,M,d) = jf_ordered_tensor.shape
        weighted_features = np.tensordot(jf_ordered_tensor, weight, axes=(2,0))
        sum1 = np.sum(np.sum(weighted_features, axis=1))
        ranked_features = np.exp(weighted_features)
        logsums = np.zeros([N,M], dtype=np.float64)
        for i1 in range(0,M):
            logsums[:,i1] = np.log(np.sum(ranked_features[:, i1:], axis=1))
    
        sum2 = np.sum(np.sum(logsums, axis=1))
        loglikelihood = sum1 - sum2
        nll = -loglikelihood
        return nll

    def __repr__(self):
        class_name = type(self).__name__
        return '{}(Trained: {})'.format(class_name, self.model is not None)
    
    def __str__(self):
        class_name = type(self).__name__
        return '{}'.format(class_name)+' with parameters: '+str(self.model)