# -*- coding: utf-8 -*-
"""
Utility functions for dyad ranking / PyDR

(2017/1)
@author: Dirk Schaefer
"""
#%%
from __future__ import division
import numpy as np
#%%
def convert_orderingvec_to_rankingvec(ordering):
    '''
        Converts an ordering vector to a ranking vector (of dim M).
        Elements of the vectors are natural numbers 1..M.
        
        Args:
            ordering - numpy 1d array
            
        Returns:
            ranking - numpy 1d array
    '''
    M = ordering.size
    ranking = -1 * np.ones(M, dtype=np.int32)
    for i1 in range(0,M):
        item_id = ordering[i1]
        if item_id > -1:
            ranking[item_id-1] = (i1+1)
    return ranking

#%%
def convert_orderingmat_to_rankingmat(Orderings):
    '''
        Converts an NxM matrix of N-many ordering row vectors into a matrix of
        rankings.
        
        Args:
            Orderings - numpy ndarray of natural numbers (1..M)
            
        Returns:
            Rankings - numpy ndarry of natural numbers (1..M)
    '''
    (Rows,Cols) = Orderings.shape    
    Rankings = -1 * np.ones((Rows, Cols))
    for i1 in range(0,Rows):
        Rankings[i1,:] = convert_orderingvec_to_rankingvec(Orderings[i1,:])
    return Rankings

#%%
def convert_rankingvec_to_orderingvec(ranking):
    '''
        Converts a ranking vector to an ordering vector (of dim M).
        Elements of the vectors are natural numbers 1..M.
        
        Args:
            ranking - numpy 1d array
            
        Returns:
            ordering - numpy 1d array
            
        Examples:
            Example 1 (complete ranking of length 4)
            R = (3,1,4,2) corresponds to O = (2,4,1,3)

            Example 2 (incomplete ranking)
            R = (-1,1,3,2) corresponds to O = (2,4,3,-1)
    '''
    M = ranking.size
    ordering = -1 * np.ones(M, dtype=np.int32)
    for i1 in range(0,M):
        r_pos = ranking[i1] # rank position of i-th label
        if r_pos > -1:
            ordering[r_pos-1] = int(i1+1)
    return ordering

#%%
def convert_rankingmat_to_orderingmat(Rankings):
    '''
        Converts an NxM matrix of N-many ranking row vectors into a matrix of
        orderings.
        
        Args:
            Rankings - numpy ndarray of natural numbers (1..M)
            
        Returns:
            Orderings - numpy ndarry of natural numbers (1..M)
    '''
    (Rows,Cols) = Rankings.shape    
    Orderings = -1 * np.ones((Rows, Cols), dtype=np.int64)
    for i1 in range(0,Rows):
        Orderings[i1,:] = convert_rankingvec_to_orderingvec(Rankings[i1,:])
    return Orderings

#%%
def invert_rankingmat(Rankings):
    '''
        Inverts an NxM (row-structured) ranking matrix.
        It assigns each value of a row its maximum minus that value + 1.
        Incomplete rankings are supported, i.e. existing -1 values are ignored.
        
        Args: 
            Rankings - numpy ndarray 
            
        Returns:
            InvRankings - numpy ndarray
        
        Note:
            This function converts rankings into relevance scores.
            By that it can be used for CGKronRLS:
                - given an ordering,
                - convert it to a ranking
                - convert it to a relevance score
                - rescale it btw [0,1]
    '''
    (nRows, nCols) = Rankings.shape
    Relevance = -1.*np.ones((nRows, nCols))
    for i1 in range(0, nRows):
        row = Rankings[i1,:]
        rmax = max(row)
        for i2 in range(0, nCols):
            if Rankings[i1,i2] > -1:
                Relevance[i1,i2] = (rmax-Rankings[i1,i2]+1) 
    return Relevance

#%%
def convert_relevancescorevec_to_rankingvec(RelevanceVec):
    '''
        Given a vector of relevance scores, expressed as natural
        numbers 1..M (e.g. associated to a common QID) of M query-document vectors,
        this function converts relevance scores (where a high score corresponds to a top rank)
        into a ranking vector.
    '''
    n = RelevanceVec.size
    temp_ordering = RelevanceVec.argsort()[::-1][:n]+1
    return convert_orderingvec_to_rankingvec(temp_ordering)
    
#%%
def convert_relevancescoresmat_to_rankingmat(RelevanceMat):
    '''
        Given a matrix of relevance vectors in rows.
        This function converts the matrix into a matrix where the 
        row vectors are rankings.
        
        This function can be used to convert testlabels (from L2R format)
        into rankings for evaluation purposes. 
    '''
    (nRows, nCols) = RelevanceMat.shape
    RankingMat = np.zeros((nRows,nCols))
    for i1 in range(0, nRows):
         RankingMat[i1,:] = convert_relevancescorevec_to_rankingvec(RelevanceMat[i1,:])
    return RankingMat
    
#%%
def kendallstau_on_rankingvec(r1, r2):
    '''
        Calculates Kendall's tau on two rankings (or likewise permutations).
        
        Args:
            r1, r2 - list of M integers
            
        Returns:
            tau - a scalar
    '''
    M = r1.size
    C = 0
    D = 0
    for i1 in range(0,M):
       for i2 in range(i1+1, M):
           p = (r1[i1]-r1[i2]) * (r2[i1]-r2[i2])
           if p>=0:
               C = C + 1
           else:
               D = D + 1
    denom = M * (M-1) / 2
    return (float(C)-float(D)) / float(denom)

#%%
def kendallstau_on_rankingmat(R1,R2):
    '''
        Calculates the Kendall's tau Distances between row vectors of R1 and R2.
    
        Args:     
            R1 and R2 must be matrices with the same dimensionality.
            
        Returns:
            mKtau - mean Ktau value
            Ktaus - vector of ktau value
    '''
    (nRows, nCols) = R1.shape
    Ktaus = np.zeros(nRows)
    for i1 in range(0, nRows):
         Ktaus[i1] = kendallstau_on_rankingvec(R1[i1,:],R2[i1,:])
    return (Ktaus.mean(), Ktaus)

#%%
def rescale_rankingmat(Rankings):
    '''
        Rescales the values of a possibly incomplete ranking matrix to the number range [0,1].
        It treats each row of the Ranking matrix separatly.
        
        Args:
            Rankings - NxM numpy ndarray of natural numbers, where -1 denotes a missing value.
            
        Returns:
            Rescaled - NxM numpy ndarray matrix which contains values within [0,1] or -1 as missing value.
        
        Note:
            This function could be used in conjunction with CgKronRLS.
    '''
    (nRows, nCols) = Rankings.shape
    Rescaled = -1.*np.ones((nRows, nCols))
    for i1 in range(0, nRows):
        row = Rankings[i1,:]
        rmax = max(row)
        rmin = 1;
        Denom = rmax-rmin
        for i2 in range(0, nCols):
            if Rankings[i1,i2] > -1:
                Rescaled[i1,i2] = (Rankings[i1,i2]-rmin)/Denom  
    return Rescaled            

def create_contextualized_concat_orderings(X, Y, Y_orderings):
    ''' Creates a list of lists from X and Y feature vectors.
        
        This method concatenates X and Y pairs and order them
        according to the matrix Y_orderings.
        
        Args:
            X : N x p list of lists / np array - instance feature vectors in rows.
            Y : M x q list of list - label feature vectors in rows.
            Y_orderings : list of N (potentially incomplete) ordering lists
            
        Returns:
            list of lists : orderings of concatenated feature vectors.
            
        Examples:
            This function can be used for preparing PLNet inputs. E.g. 
            for label ranking data:
            Z = create_contextualized_concat_orderings(X.tolist(), ...
            np.eye(M, dtype=np.int32).tolist() , ...
            dp.utils.convert_rankingmat_to_orderingmat(R))
    '''
    N = len(Y_orderings)
    Z = []
    for i1 in range(N):
        Z.append([X[i1] + Y[i0-1] for i0 in Y_orderings[i1]])
    return Z

#%%
def get_kronecker_feature_map_tensor(XFeat, YFeat):
    """
    Produces a tensor of the format rows x cols x features.
    This produces a contextualized dyad ranking tensor.
    
    Args:
        XFeat : ndarray (matrix Nxp)
        YFeat : ndarry (matrix Mxq)
    
    Returns:
        jf_tensor : ndarray (tensor N x M x (p*q))
            
    """
    (N,p) = XFeat.shape
    (M,q) = YFeat.shape

    kronTensor = np.zeros([N,M, p*q], dtype=np.float64)

    for i1 in range(0,N):
        for i2 in range(0,M):
            kfeat = np.kron(XFeat[i1,:],YFeat[i2,:])
            kronTensor[i1,i2,:] = kfeat
    
    return kronTensor
    