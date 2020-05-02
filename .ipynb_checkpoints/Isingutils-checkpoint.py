#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Lasso



def random_normal(nrow, ncol):
    W = np.zeros(ncol**2)
    W = W.reshape(ncol, -1)
    for i in range(ncol):
        for j in range(i, ncol):
            r = abs(round(np.random.normal(0, 2), 2))
            W[i, j] = r
            W[j, i] = r
    return(np.random.multivariate_normal(mean=[0] * ncol, cov = W, size = nrow))

class Ising:
    def __init__(self, W, u):
        #assert(W.shape[0] == W.shape[1] and W.shape[0] == u.shape[0])
        #assert(np.allclose(W, W.T, atol=1e-8))
        self.W = W
        self.u = u
        self.d = W.shape[0]
        
    @staticmethod
    def gibbs_sampling(model, n):
        X = np.array([+1 if np.random.rand() < .5 else -1 for i in range(model.d)])
        samples = [np.copy(X)]
        for i in range(2*n + 99):
            for j in range(model.d):
                p = model.conditonal(j, X)
                X[j] = +1 if np.random.rand() < p else -1
            samples.append(np.copy(X))
        return np.array(samples[100::2])
    
    
    @staticmethod
    def get_sample(W, n):
        u = np.zeros(W.shape[0])
        ising_model = Ising(W, u)
        return Ising.gibbs_sampling(ising_model, n)
    
    @staticmethod
    def random_coupling(ncol):
        W = np.zeros(ncol**2)
        W = W.reshape(ncol, -1)
        for j in range(int(30)):
            for i in range(ncol):
                x = np.random.choice(ncol, 1, replace=False)
                y = np.random.choice(i+1, 1, replace=False)
                r = round(np.random.normal(0, 1), 2)
                W[x, y] = r
                W[y, x] = r 
        return(W)
    
    @staticmethod
    def save_ising(Z, W, name_theta = "Theta.csv", name_z = "Z.csv"):
        W_df = pd.DataFrame(W)
        W_df.to_csv(name_theta, index=False, index_label=False)
        Z_df = pd.DataFrame(Z)
        Z_df.to_csv(name_z, index=False, index_label=False)
        
    def conditonal(self, i, X):
        def sigmoid(x):
            return 1. / (1 + np.exp(-x))
        tmp = self.W[i, :].dot(X)
        return sigmoid(2 * (tmp + self.u[i]))

    def energy(self, X):
        return -X.dot(self.W).dot(X) - X.dot(self.u)



class Ising_Data:
    
    def __init__(self, Z):
        self.Z = Z
        
    def predict_cluster(self, k):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(self.Z)
        return(kmeans.predict(self.Z))
    
    def reduce_cluster(self, k):
        predicted_clusters = self.predict_cluster(k)
        Z_reduced = np.zeros(shape = (self.Z.shape[0], k)) - 1
        for i in range(self.Z.shape[0]):
            Z_reduced[i, predicted_clusters[i]] = 1
        states = predicted_clusters
        TM = np.zeros((len(np.unique(states)), len(np.unique(states))))
        for s in np.unique(states):
            c = 0
            n_transitions = sum(states[1:] == s)
            for i, ts in enumerate(states[1:]):
                if ts == s:
                    TM[states[1:][i-1], ts] += 1/n_transitions
        TM = TM.T
        return(Z_reduced, TM)
    
    @staticmethod
    def joint_coupling(Z, Y):
        return(Z.T.dot(Y) / Z.shape[1])
        
    

        
        
        
        

