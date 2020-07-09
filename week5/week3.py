# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:52:00 2020

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

os.chdir(r"C:\Users\jlo\Documents\Summer20\HUANG\Data")
pp_donorA = "donorA/filtered_matrices_mex/hg19/write/donorA_pp.h5ad"
pp_donorB = "donorB/filtered_matrices_mex/hg19/write/donorB_pp.h5ad"

sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor = 'white')

DonorA = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\donorA\filtered_matrices_mex\hg19',var_names = 'gene_symbols', cache = True)
DonorB = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\donorB\filtered_matrices_mex\hg19',var_names = 'gene_symbols', cache = True)

sc.pl.highest_expr_genes(DonorA, n_top=20,)
sc.pl.highest_expr_genes(DonorB, n_top=20,)
sc.pp.filter_cells(DonorA, min_genes = 200)
sc.pp.filter_cells(DonorB, min_genes = 200)
sc.pp.filter_genes(DonorA, min_cells=5)
sc.pp.filter_genes(DonorB, min_cells=5)
DonorA.var['mt'] = DonorA.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error
DonorB.var['mt'] = DonorB.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error
sc.pp.calculate_qc_metrics(DonorA, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)
sc.pp.calculate_qc_metrics(DonorB, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)
sc.pl.violin(DonorA, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .4, multi_panel = True)
sc.pl.violin(DonorB, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .4, multi_panel = True)

sc.pl.scatter(DonorA, x = 'total_counts', y = 'pct_counts_mt')
sc.pl.scatter(DonorB, x = 'total_counts', y = 'pct_counts_mt')#visualize - note that the mt percentage is on average higher for this dataset, and there appears to be a clear elbow starting at around 18
DonorA = DonorA[DonorA.obs.pct_counts_mt<5, :]
DonorB = DonorB[DonorB.obs.pct_counts_mt<6, :]  #filter
sc.pl.scatter(DonorA, x = 'total_counts', y = 'pct_counts_mt')#visualize after
sc.pl.scatter(DonorB, x = 'total_counts', y = 'pct_counts_mt')

sc.pl.scatter(DonorA, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorB, x = 'total_counts', y = 'n_genes_by_counts')#visualize
DonorA = DonorA[DonorA.obs.n_genes_by_counts <2000, :]
DonorB = DonorB[DonorB.obs.n_genes_by_counts <2000, :]
sc.pl.scatter(DonorA, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorB, x = 'total_counts', y = 'n_genes_by_counts')

sc.pp.normalize_total(DonorA)
sc.pp.normalize_total(DonorB)

sc.pp.log1p(DonorA)
sc.pp.log1p(DonorB)

sc.pp.highly_variable_genes(DonorA, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pp.highly_variable_genes(DonorB, min_mean = .0125, max_mean = 3, min_disp = .5)

sc.pl.highly_variable_genes(DonorA)
sc.pl.highly_variable_genes(DonorB)
DonorA = DonorA[:, DonorA.var.highly_variable]
DonorB = DonorB[:, DonorB.var.highly_variable]
sc.pp.regress_out(DonorA, ['total_counts', 'pct_counts_mt'])
sc.pp.regress_out(DonorB, ['total_counts', 'pct_counts_mt'])

##scale to unit variance
sc.pp.scale(DonorA, max_value = 10)
sc.pp.scale(DonorB, max_value = 10)

sc.tl.pca(DonorA, svd_solver='arpack')
sc.tl.pca(DonorB, svd_solver='arpack')
#save
DonorA.write(pp_donorA)
DonorB.write(pp_donorB)

#neighborhood graph
neighbors = 5
sc.pp.neighbors(DonorA_sample, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(DonorB_sample, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.tl.umap(DonorA_sample)
sc.tl.umap(DonorB_sample)



#compute R_a and R_b accordiung to Wang 2009
##DonorA_NN = NearestNeighbors(n_neighbors = 50, algorithm = 'ball_tree', metric = "euclidean").fit(DonorA.X)
#distances_A, indices_A = DonorA_NN.kneighbors(DonorA.X)
# =============================================================================
# data = DonorA.X
# obs = list(range(0,len(DonorA.X)))
# df = pd.DataFrame(data, columns = DonorA.var_names, index = obs)
# DonorA_DM = distance_matrix(df.values, df.values)
# DonorA_R = [0]*DonorA_DM.shape[0]
# permutations = math.factorial(neighbors-1)
# for i in range(0,DonorA_DM.shape[0]):
#     indices = find(DonorA.obsp['distances'][i])[1]
#     DonorA_R[i] = [0]*permutations
#     permuted_indices = list(itertools.permutations(indices))
#     for p in range(0,permutations):
#         permuted_indices[p] = np.insert(permuted_indices[p], 0, i)  
#         DonorA_R[i][p] = np.zeros(shape = (neighbors,neighbors))
#         base_row = DonorA_DM[i,permuted_indices[p]]
#         DonorA_R[i][p][0,:] = base_row 
#         for j in range(1,neighbors):
#             for k in range(0,neighbors):
#                 DonorA_R[i][p][j,k] = DonorA_DM[permuted_indices[p][j], permuted_indices[p][k]]
# 
# =============================================================================
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
   
# =============================================================================
# DonorB_R = [0]*DonorB.obsp['distances'].shape[0]
# permutations = math.factorial(neighbors-1)
# for i in range(0,DonorB.obsp['distances'].shape[0]-1):
#     indices = find(DonorB.obsp['distances'][i])[1]
#     DonorB_R[i] = [0]*permutations
#     permuted_indices = list(itertools.permutations(indices))
#     for p in range(0,permutations-1):
#         permuted_indices[p] = np.insert(permuted_indices[p], 0, 0)  
#         DonorB_R[i][p] = np.zeros(shape = (neighbors,neighbors))
#         base_row = DonorB.obsp['distances'][i,permuted_indices[p]]
#         DonorB_R[i][p][0,:] = base_row.toarray() 
#         for j in range(1,neighbors-1):
#             for k in range(0,neighbors-1):
#                 DonorB_R[i][p][j,k] = DonorB.obsp['distances'][permuted_indices[p][j], permuted_indices[p][k]]
# 
# =============================================================================
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

A_transformed = np.matmul(A.transpose(),DonorA_sample.X.transpose())
B_transformed = np.matmul(B.transpose(),DonorB_sample.X.transpose())

AB_integrated = np.concatenate((A_transformed.transpose(), B_transformed.transpose()), axis = 0)

#artificially change labels for donor B
newvars = [str(i) for i in range(1,(len(DonorB.var_names)+1))]
DonorB.var_names = newvars
#merge datasets
Donors_merged= DonorA.concatenate(DonorB, join = 'outer')






sc.pl.highest_expr_genes(Donors_merged, n_top=20,)
sc.pp.filter_cells(Donors_merged, min_genes = 200)
sc.pp.filter_genes(Donors_merged, min_cells=5)
Donors_merged.var['mt'] = Donors_merged.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error
sc.pp.calculate_qc_metrics(adata_merged, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)
sc.pl.violin(adata_merged, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .4, multi_panel = True)


scanpy.pp.neighbors(DonorA)
scanpy.pp.neighbors(DonorB)