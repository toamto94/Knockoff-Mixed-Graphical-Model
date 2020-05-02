#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Lasso
from Isingutils import Ising_Data


class Ising_Normal_Knockoffs:
    
    def __init__(self, Z, Y, theta, C_zy):
        self.theta = theta
        self.C_zy = C_zy
        self.n_ising = theta.shape[1]
        self.n_normal = C_zy.shape[1]
        self.Z = Z
        self.Y = Y
        
        
    def __energy(self, i, z_i, z, zt, y):
        e = 0
        for j in range(self.n_ising):
            if j != i:
                e += z_i * zt[j] * self.theta[i, j]
        for j in range(self.n_ising):
            e += z_i * z[j] * self.theta[i, j]

        #for j in range(self.n_normal):
        #    e += z_i * y[j] * self.C_zy[i, j]
        return(np.exp(e))
    
    def __predict_sample(self, i, z, zt, y):
        p_one = self.__energy(i=i, z=z, zt=zt, y=y, z_i=1)
        p_minus_one = self.__energy(i=i, z=z, zt=zt, y=y, z_i=-1)
        u = np.random.uniform()
        prob = p_minus_one / (p_one + p_minus_one)
        if u <= prob:
            return(-1)
        else:
            return(1)
         
    def sample_row(self, z, y, zt=None):
        if zt is None:
            zt=np.random.choice(a=[-1, 1], replace=True, size=self.n_ising)
        for k in range(1):
            for i in range(self.n_ising):
                zt[i] = self.__predict_sample(i=i, z=z, zt=zt, y=y)
        return(zt)
    
    def sample_knockoffs(self, k=None, return_clusters=False):
        Zt = np.zeros_like(self.Z)
        Yt = np.zeros_like(self.Y)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.Z)
        clusters = kmeans.predict(self.Z)
        first_z = self.sample_row(self.Z[0, :], self.Y[0, :], self.Z[0, :])
        first_pred = kmeans.predict(first_z.reshape(1, -1))[0] 
        filtered_cov = np.cov(self.Y[np.argwhere(clusters == first_pred).reshape(-1), :].T)
        first_y = np.random.multivariate_normal(mean=[0] * len(filtered_cov), cov=filtered_cov, size=1)
        
        Zt[0, :] = first_z
        Yt[0, :] = first_y
        for i in range(1, self.Z.shape[0]):
            Zt[i, :] = self.sample_row(self.Z[i, :], self.Y[i, :], Zt[i-1, :])
            filtered_cov = np.cov(self.Y[np.argwhere(clusters == kmeans.predict(Zt[i, :].reshape(1, -1))[0]).reshape(-1), :].T)
            Yt[i, :] = np.random.multivariate_normal(mean=[0] * len(filtered_cov), cov=filtered_cov, size=1)
            if i % 100 == 0:
                print(str(i) + " knockoffs generated ...")
        if not return_clusters:
            return(Zt, Yt)
        else:
            return(Zt, Yt, clusters)
        
        
class Knockoff_Mixed_Graphical_Model:
    
    def __init__(self):
        self.Z = None
        self.Y = None
        self.ne_Z = None
        self.ne_Y = None
        self.vertices = None
    
    def __split(self, X, i):
        return(X[:, i], np.delete(X, i, axis=1))
    
    def __get_indices_fs(self, x, i):
        first = np.arange(i)
        end = np.arange(i + 1, len(x) + 1)
        return(np.hstack((first, end)))
    
    def __lasso_coefficient_difference(self, x, x_tilde):
        return(abs(x) - abs(x_tilde))
    
    def fit(self, Z, Y, k, feature_statistics_fnc = "lcd"):
        
        self.Z = Z
        self.Y = Y
        
        ne_Z = []
        ncol_z = Z.shape[1]
        ncol_y = Y.shape[1]
        for i in range(ncol_z):
            Z_i, Z_mi = self.__split(Z, i)
            Z_mi_cov = Z_mi.T.dot(Z_mi)/len(Z_mi)

            C = Ising_Data.joint_coupling(Z_mi, Y)
            INK = Ising_Normal_Knockoffs(Z_mi, Y, Z_mi_cov, C)
            Z_tilde, Y_tilde, clusters = INK.sample_knockoffs(k = k, return_clusters=True)

            X = np.hstack((Z_mi, Z_tilde, Y, Y_tilde))
            LR = LogisticRegression(penalty='l1', solver='liblinear')
            LR.fit(X=X, y=Z_i)
            coef = [x[0] for x in LR.coef_.T]

            ne_i = []
            for z in range(Z_mi.shape[1]):
                fs = self.__lasso_coefficient_difference(coef[z], coef[z + Z_mi.shape[1]])
                ne_i.append(fs)
            for y in range(Y.shape[1]):
                fs = self.__lasso_coefficient_difference(coef[y + Z_mi.shape[1]], coef[y + Z_mi.shape[1] + Y.shape[1]])
                ne_i.append(fs)
            ne_Z.append(ne_i)
        ne_Z = np.array(ne_Z)
        
        
        ne_Y = []
        for i in range(ncol_y):
            Y_i, Y_mi = self.__split(Y, i)
            Z_cov = Z.T.dot(Z)/len(Z)

            C = Ising_Data.joint_coupling(Z, Y_mi)
            INK = Ising_Normal_Knockoffs(Z, Y_mi, Z_cov, C)
            Z_tilde, Y_tilde, clusters = INK.sample_knockoffs(k = k, return_clusters=True)

            X = np.hstack((Z, Z_tilde, Y_mi, Y_tilde))
            LASSO = Lasso()
            LASSO.fit(X=X, y=Y_i)
            coef = LASSO.coef_

            ne_i = []
            for z in range(Z.shape[1]):
                fs = self.__lasso_coefficient_difference(coef[z], coef[z + Z.shape[1]])
                ne_i.append(fs)
            for y in range(Y_mi.shape[1]):
                fs = self.__lasso_coefficient_difference(coef[y + Z.shape[1]], coef[y + Z.shape[1] + Y_mi.shape[1]])
                ne_i.append(fs)
            ne_Y.append(ne_i)
        ne_Y = np.array(ne_Y)
        
        self.ne_Z = ne_Z
        self.ne_Y = ne_Y
        
        
    def merge_neighborhoods(self, strategy="union", th = 0.1):
        
        assert(self.Z is not None)
        assert(self.Y is not None)
        assert(self.ne_Z is not None)
        assert(self.ne_Y is not None)
    
        ne_disc = []
        for i in range(self.Z.shape[1]):
            discs_indices = self.__get_indices_fs(self.ne_Z[i, :], i)[np.logical_or(self.ne_Z[i, :] >= th , self.ne_Z[i, :] <= -th)]
            ne_disc.append(discs_indices)
        for i in range(self.Y.shape[1]):
            discs_indices = self.__get_indices_fs(self.ne_Y[i, :], i + self.Z.shape[1])[np.logical_or(self.ne_Y[i, :] >= th , self.ne_Y[i, :] <= -th)]
            ne_disc.append(discs_indices)
            
        if strategy == "union":
            ne_disc_union = ne_disc.copy()
            for i, col in enumerate(ne_disc_union):
                for j in col:
                    if i in ne_disc_union[j]:
                        pass
                    else:
                        ne_disc_union[i] = ne_disc_union[i][ne_disc_union[i] != j]
                        
        self.vertices = ne_disc_union
        
    def get_vertices(self):
        assert(self.vertices is not None)
        return(self.vertices)
        
    

        
        
        
        

