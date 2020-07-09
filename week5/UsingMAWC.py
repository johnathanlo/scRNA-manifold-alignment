# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:22:40 2020

@author: jlo
"""

import itertools
import math
import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
from harmony import harmonize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, linalg
from scipy.spatial import distance_matrix
from scipy.linalg import block_diag
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

data = DonorA_sample.X
obs = list(range(0,len(DonorA_sample.X)))
df = pd.DataFrame(data, columns = DonorA_sample.var_names, index = obs)
DonorA_DM = distance_matrix(df.values, df.values)
DonorA_R = [0]*DonorA_DM.shape[0]
permutations = math.factorial(neighbors-1)
for i in range(0,DonorA_DM.shape[0]):
    indices = find(DonorA_sample.obsp['distances'][i])[1]
    DonorA_R[i] = [0]*permutations
    permuted_indices = list(itertools.permutations(indices))
    for p in range(0,permutations):
        permuted_indices[p] = np.insert(permuted_indices[p], 0, i)  
        DonorA_R[i][p] = np.zeros(shape = (neighbors,neighbors))
        base_row = DonorA_DM[i,permuted_indices[p]]
        DonorA_R[i][p][0,:] = base_row 
        for j in range(1,neighbors):
            for k in range(0,neighbors):
                DonorA_R[i][p][j,k] = DonorA_DM[permuted_indices[p][j], permuted_indices[p][k]]

 
data = DonorB_sample.X
obs = list(range(0,len(DonorB_sample.X)))
df = pd.DataFrame(data, columns = DonorB_sample.var_names, index = obs)
DonorB_DM = distance_matrix(df.values, df.values)
DonorB_R = [0]*DonorB_DM.shape[0]
permutations = math.factorial(neighbors-1)
for i in range(0,DonorB_DM.shape[0]):
    indices = find(DonorB_sample.obsp['distances'][i])[1]
    DonorB_R[i] = [0]*permutations
    permuted_indices = list(itertools.permutations(indices))
    for p in range(0,permutations):
        permuted_indices[p] = np.insert(permuted_indices[p], 0, i)  
        DonorB_R[i][p] = np.zeros(shape = (neighbors,neighbors))
        base_row = DonorB_DM[i,permuted_indices[p]]
        DonorB_R[i][p][0,:] = base_row 
        for j in range(1,neighbors):
            for k in range(0,neighbors):
                DonorB_R[i][p][j,k] = DonorB_DM[permuted_indices[p][j], permuted_indices[p][k]]

DistAB = np.zeros(shape = (DonorA_DM.shape[0], DonorB_DM.shape[0]))
for i in range(0, DonorA_DM.shape[0]):
        for j in range(0, DonorB_DM.shape[0]):
            mindists = list()
            for p in range(0, permutations):
                k1 = np.matmul(DonorA_R[i][0].transpose(), DonorB_R[j][p]).trace()/np.matmul(DonorA_R[i][0].transpose(), DonorA_R[i][0]).trace()
                k2 = np.matmul(DonorB_R[j][p].transpose(), DonorA_R[i][0]).trace()/np.matmul(DonorB_R[j][p].transpose(), DonorB_R[j][p]).trace()
                dist1_h = np.linalg.norm(DonorB_R[j][p] - k1*DonorA_R[i][0])
                dist2_h = np.linalg.norm(DonorA_R[i][0] - k2*DonorB_R[j][p])
                mindists.append(min(dist1_h, dist2_h))
                if p == permutations-1:
                    DistAB[i][j] = min(mindists)

np.savetxt('DistAB.csv', DistAB, delimiter = ',')
W = np.zeros(shape = (DonorA_DM.shape[0], DonorB_DM.shape[0]))
delta = 1
for i in range(0, DonorA_DM.shape[0]):
        for j in range(0, DonorB_DM.shape[0]):
            W[i][j] = math.exp(-DistAB[i][j]/(delta**2))
            
W_A = 1/(1+DonorA_DM)
D_A = np.zeros(shape = (DonorA_DM.shape[0], DonorA_DM.shape[0]))
for i in range(0, DonorA_DM.shape[0]):
    D_A[i][i] = sum(W_A[i])
L_A = D_A - W_A

W_B = 1/(1+DonorB_DM)
D_B = np.zeros(shape = (DonorB_DM.shape[0], DonorB_DM.shape[0]))
for i in range(0, DonorB_DM.shape[0]):
    D_B[i][i] = sum(W_B[i])
L_B = D_B - W_B

Omega1 = np.zeros(shape = (DonorA_DM.shape[0], DonorA_DM.shape[0]))
for i in range(0, DonorA_DM.shape[0]):
    Omega1[i][i] = sum(W[i])
    
Omega2 = np.zeros(shape = (DonorA_DM.shape[0], DonorB_DM.shape[0]))
for i in range(0, DonorA_DM.shape[0]):
    for j in range(0, DonorB_DM.shape[0]):
        Omega2[i][j] = W[i][j]

Omega3 = np.zeros(shape = (DonorB_DM.shape[0], DonorA_DM.shape[0]))
for i in range(0, DonorB_DM.shape[0]):
    for j in range(0, DonorA_DM.shape[0]):
        Omega3[i][j] = W[j][i]
        
Omega4 = np.zeros(shape = (DonorB_DM.shape[0], DonorB_DM.shape[0]))
for i in range(0, DonorB_DM.shape[0]):
    Omega4[i][i] = sum(W[:,i])

Z = block_diag(DonorA_sample.X.transpose(), DonorB_sample.X.transpose())
D = block_diag(D_A, D_B)
mu = .5
L_11 = L_A + mu*Omega1
L_12 = -mu*Omega2
L_21 = -mu*Omega3
L_22 = L_B + mu*Omega4 
L = np.block([[L_11, L_12], [L_21, L_22]])

ZLZ = np.matmul(Z, L)
ZLZ = np.matmul(ZLZ, Z.transpose())
ZDZ = np.matmul(Z,D)
ZDZ = np.matmul(ZDZ, Z.transpose())

lambdas, gammas =  linalg.eigs(ZLZ, M = ZDZ, which = 'SR')

A = gammas[0:DonorA_sample.shape[1],:]
B = gammas[DonorA_sample.shape[1]:, :]

A_transformed = np.matmul(A.real.transpose(),DonorA_sample.X.transpose())
B_transformed = np.matmul(B.real.transpose(),DonorB_sample.X.transpose())

AB_integrated = AB_integrated = [A_transformed.transpose(), B_transformed.transpose()]
################visualize############
data = [DonorA_sample.X, DonorB_sample.X]
dataset_num = len(data)
data_integrated = AB_integrated
styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 
datatype = None
embedding = []
dataset_xyz = []
for i in range(dataset_num):
    dataset_xyz.append("data{:d}".format(i+1))
    embedding.append(PCA(n_components=2).fit_transform(data[i]))
   
fig = plt.figure()
if datatype is not None:
    for i in range(dataset_num):
        plt.subplot(1,dataset_num,i+1)
        for j in set(datatype[i]):
            index = np.where(datatype[i]==j) 
            plt.scatter(embedding[i][index,0], embedding[i][index,1], c=styles[j], s=5.)
        plt.title(dataset_xyz[i])
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.legend()
else:
    for i in range(dataset_num):
        plt.subplot(1,dataset_num,i+1)
        plt.scatter(embedding[i][:,0], embedding[i][:,1],c=styles[i], s=5.)
        plt.title(dataset_xyz[i])
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.legend()

plt.tight_layout()

data_all = np.vstack((data_integrated[0], data_integrated[1]))
for i in range(2, dataset_num):
    data_all = np.vstack((data_all, data_integrated[i]))

embedding_all = PCA(n_components=2).fit_transform(data_all)

tmp = 0
num = [0]
for i in range(dataset_num):
    num.append(tmp+np.shape(data_integrated[i])[0])
    tmp += np.shape(data_integrated[i])[0]

embedding = []
for i in range(dataset_num):
    embedding.append(embedding_all[num[i]:num[i+1]])

color = [[1,0.5,0], [0.2,0.4,0.1], [0.1,0.2,0.8], [0.5, 1, 0.5], [0.1, 0.8, 0.2]]
# marker=['x','^','o','*','v']
    
fig = plt.figure()
if datatype is not None:
    plt.subplot(1,2,1)
    for i in range(dataset_num):
        plt.scatter(embedding[i][:,0], embedding[i][:,1], c=color[i], label='data{:d}'.format(i+1), s=5., alpha=0.8)
    plt.title('Integrated Embeddings')
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.legend()

    plt.subplot(1,2,2)
    for i in range(dataset_num):  
        for j in set(datatype[i]):
            index = np.where(datatype[i]==j) 
            if i < dataset_num-1:
                plt.scatter(embedding[i][index,0], embedding[i][index,1], c=styles[j], s=5., alpha=0.8)
            else:
                plt.scatter(embedding[i][index,0], embedding[i][index,1], c=styles[j], s=5., alpha=0.8)  
    plt.title('Integrated Cell Types')
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.legend()

else:

    for i in range(dataset_num):
        plt.scatter(embedding[i][:,0], embedding[i][:,1], c=styles[i], label='data{:d}'.format(i+1), s=5., alpha=0.8)
    plt.title('Integrated Embeddings')
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.legend()

plt.tight_layout()
plt.show()