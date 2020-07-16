# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:36:53 2020

@author: jlo
"""


import numpy as np
from scipy import sparse
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
from os import mkdir, path, remove, stat
from time import time
import subprocess
import pathlib


import os
os.environ['PYTHONHOME'] = 'C:/Program Files/Python'
os.environ['PYTHONPATH'] = 'C:/Program Files/Python/lib/site-packages'
os.environ['R_HOME'] = 'C:/Program Files/R/R-3.5.1'
os.environ['R_USER'] = 'C:/Program Files/Python/Lib/site-packages/rpy2'

# importing rpy2 now throws no errors
import rpy2.robjects as ro
import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR) # Ignore R warning messages
import rpy2.robjects as ro
import anndata2ri
from scipy.sparse import csgraph

import gc

def checkAdata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')

def checkBatch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        print(f'Object contains {obs[batch].nunique()} batches.')

def checkHVG(hvg, adata_var):
    if type(hvg) is not list:
        raise TypeError('HVG list is not a list')
    else:
        if not all(i in adata_var.index for i in hvg):
            raise ValueError('Not all HVGs are in the adata object')

def checkSanity(adata, batch, hvg):
    checkAdata(adata)
    checkBatch(batch, adata.obs)
    if hvg is not None:
        checkHVG(hvg, adata.var)


def splitBatches(adata, batch, hvg= None):
    split = []
    if hvg is not None:
        adata = adata[:, hvg]
    for i in adata.obs[batch].unique():
        split.append(adata[adata.obs[batch]==i].copy())
    return split

def merge_adata(adata_list, sep='-'):
    """
    merge adatas from list and remove duplicated obs and var columns
    """
    
    if len(adata_list) == 1:
        return adata_list[0]
    
    adata = adata_list[0].concatenate(*adata_list[1:], index_unique=None, batch_key='tmp')
    del adata.obs['tmp']

    if len(adata.obs.columns) > 0:
        # if there is a column with separator
        if sum(adata.obs.columns.str.contains(sep)) > 0:
            columns_to_keep = [name.split(sep)[1] == '0' for name in adata.var.columns.values]
            clean_var = adata.var.loc[:, columns_to_keep]
        else:
            clean_var = adata.var
            
    if len(adata.var.columns) > 0:
        if sum(adata.var.columns.str.contains(sep)) > 0:
            adata.var = clean_var.rename(columns={name : name.split('-')[0] for name in clean_var.columns.values})
        
    return adata


def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        adata.X = adata.X.todense()
        
def kBET_single(matrix, batch, type_ = None, k0 = 10, knn=None, subsample=0.5, heuristic=True, verbose=False):
    """
    params:
        matrix: expression matrix (at the moment: a PCA matrix, so do.pca is set to FALSE
        batch: series or list of batch assignemnts
        subsample: fraction to be subsampled. No subsampling if `subsample=None`
    returns:
        kBET p-value
    """
        
    anndata2ri.activate()
    ro.r("library(kBET)")
    
    if verbose:
        print("importing expression matrix")
    ro.globalenv['data_mtrx'] = matrix
    ro.globalenv['batch'] = batch
    #print(matrix.shape)
    #print(len(batch))
    
    if verbose:
        print("kBET estimation")
    #k0 = len(batch) if len(batch) < 50 else 'NULL'
    
    ro.globalenv['knn_graph'] = knn
    ro.globalenv['k0'] = k0
    batch_estimate = ro.r(f"batch.estimate <- kBET(data_mtrx, batch, knn=knn_graph, k0=k0, plot=FALSE, do.pca=FALSE, heuristic=FALSE, adapt=FALSE, verbose={str(verbose).upper()})")
            
    anndata2ri.deactivate()
    try:
        ro.r("batch.estimate$average.pval")[0]
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        return np.nan
    else:
        return ro.r("batch.estimate$average.pval")[0]


def kBET(adata, batch_key, label_key, embed='X_pca', type_ = None,
                    hvg=False, subsample=0.5, heuristic=False, verbose=False):
    """
    Compare the effect before and after integration
    params:
        matrix: matrix from adata to calculate on
    return:
        pd.DataFrame with kBET p-values per cluster for batch
    """
    
    checkAdata(adata)
    checkBatch(batch_key, adata.obs)
    checkBatch(label_key, adata.obs)
    #compute connectivities for non-knn type data integrations
    #and increase neighborhoods for knn type data integrations
    if type_ != 'knn':
        adata_tmp = sc.pp.neighbors(adata, n_neighbors = 50, use_rep=embed, copy=True)
    else:
        #check if pre-computed neighbours are stored in input file
        adata_tmp = adata.copy()
        if 'diffusion_connectivities' not in adata.uns['neighbors']:
            if verbose:
                print(f"Compute: Diffusion neighbours.")
            adata_tmp = diffusion_conn(adata, min_k = 50, copy = True)
        adata_tmp.uns['neighbors']['connectivities'] = adata_tmp.uns['neighbors']['diffusion_connectivities']
            
    if verbose:
        print(f"batch: {batch_key}")
        
    #set upper bound for k0
    size_max = 2**31 - 1
    
    kBET_scores = {'cluster': [], 'kBET': []}
    for clus in adata_tmp.obs[label_key].unique():
        
        adata_sub = adata_tmp[adata_tmp.obs[label_key] == clus,:].copy()
        #check if neighborhood size too small or only one batch in subset
        if np.logical_or(adata_sub.n_obs < 10, 
                         len(adata_sub.obs[batch_key].cat.categories)==1):
            print(f"{clus} consists of a single batch or is too small. Skip.")
            score = np.nan
        else:
            quarter_mean = np.floor(np.mean(adata_sub.obs[batch_key].value_counts())/4).astype('int')
            k0 = np.min([70, np.max([10, quarter_mean])])
            #check k0 for reasonability
            if (k0*adata_sub.n_obs) >=size_max:
                k0 = np.floor(size_max/adata_sub.n_obs).astype('int')
           
            matrix = np.zeros(shape=(adata_sub.n_obs, k0+1))
                
            if verbose:
                print(f"Use {k0} nearest neighbors.")
            n_comp, labs = csgraph.connected_components(adata_sub.uns['neighbors']['connectivities'], 
                                                              connection='strong')
            if n_comp > 1:
                #check the number of components where kBET can be computed upon
                comp_size = pd.value_counts(labs)
                #check which components are small
                comp_size_thresh = 3*k0
                idx_nonan = np.flatnonzero(np.in1d(labs, 
                                                   comp_size[comp_size>=comp_size_thresh].index))
                #check if 75% of all cells can be used for kBET run
                if len(idx_nonan)/len(labs) >= 0.75:
                    #create another subset of components, assume they are not visited in a diffusion process
                    adata_sub_sub = adata_sub[idx_nonan,:].copy()
                    nn_index_tmp = np.empty(shape=(adata_sub.n_obs, k0))
                    nn_index_tmp[:] = np.nan
                    nn_index_tmp[idx_nonan] = diffusion_nn(adata_sub_sub, k=k0).astype('float') 
                    #need to check neighbors (k0 or k0-1) as input?   
                    score = kBET_single(
                            matrix=matrix,
                            batch=adata_sub.obs[batch_key],
                            knn = nn_index_tmp+1, #nn_index in python is 0-based and 1-based in R
                            subsample=subsample,
                            verbose=verbose,
                            heuristic=False,
                            k0 = k0,
                            type_ = type_
                            )
                else:
                    #if there are too many too small connected components, set kBET score to 1 
                    #(i.e. 100% rejection)
                    score = 1
                
            else: #a single component to compute kBET on 
                #need to check neighbors (k0 or k0-1) as input?  
                nn_index_tmp = diffusion_nn(adata_sub, k=k0).astype('float')
                score = kBET_single(
                            matrix=matrix,
                            batch=adata_sub.obs[batch_key],
                            knn = nn_index_tmp+1, #nn_index in python is 0-based and 1-based in R
                            subsample=subsample,
                            verbose=verbose,
                            heuristic=False,
                            k0 = k0,
                            type_ = type_
                            )
        
        kBET_scores['cluster'].append(clus)
        kBET_scores['kBET'].append(score)
    
    kBET_scores = pd.DataFrame.from_dict(kBET_scores)
    kBET_scores = kBET_scores.reset_index(drop=True)
    
    return kBET_scores

def diffusion_nn(adata, k, max_iterations=16):
    '''
    This function generates a nearest neighbour list from a connectivities matrix
    as supplied by BBKNN or Conos. This allows us to select a consistent number
    of nearest neighbours across all methods.
    Return:
       `k_indices` a numpy.ndarray of the indices of the k-nearest neighbors.
    '''
    ''' if 'neighbors' not in adata.uns:
        raise ValueError('`neighbors` not in adata object. '
                         'Please compute a neighbourhood graph!')
    
    if 'connectivities' not in adata.uns['neighbors']:
        raise ValueError('`connectivities` not in `adata.uns["neighbors"]`. '
                         'Please pass an object with connectivities computed!')
   '''     
    T = adata.uns['neighbors']['connectivities']

    # Row-normalize T
    T = sparse.diags(1/T.sum(1).A.ravel())*T
    
    T_agg = T**3
    M = T+T**2+T_agg
    i = 4
    
    while ((M>0).sum(1).min() < (k+1)) and (i < max_iterations): 
        #note: k+1 is used as diag is non-zero (self-loops)
        print(f'Adding diffusion to step {i}')
        T_agg *= T
        M += T_agg
        i+=1

    if (M>0).sum(1).min() < (k+1):
        raise ValueError(f'could not find {k} nearest neighbors in {max_iterations}'
                         'diffusion steps.\n Please increase max_iterations or reduce'
                         ' k.\n')
    
    M.setdiag(0)
    k_indices = np.argpartition(M.A, -k, axis=1)[:, -k:]
    
    return k_indices

# determine root cell for trajectory conservation metric
def get_root(adata_pre, adata_post, ct_key, dpt_dim=3):
    n_components, adata_post.obs['neighborhood'] = csgraph.connected_components(csgraph=adata_post.uns['neighbors']['connectivities'], directed=False, return_labels=True)
    
    start_clust = adata_pre.obs.groupby([ct_key]).mean()['dpt_pseudotime'].idxmin()
    min_dpt = np.flatnonzero(adata_pre.obs[ct_key] == start_clust)
    max_neigh = np.flatnonzero(adata_post.obs['neighborhood']== adata_post.obs['neighborhood'].value_counts().argmax())
    min_dpt = [value for value in min_dpt if value in max_neigh]
    
    # compute Diffmap for adata_post
    sc.tl.diffmap(adata_post)
    
    # determine most extreme cell in adata_post Diffmap
    min_dpt_cell = np.zeros(len(min_dpt))
    for dim in np.arange(dpt_dim):
        
        diffmap_mean = adata_post.obsm["X_diffmap"][:, dim].mean()
        diffmap_min_dpt = adata_post.obsm["X_diffmap"][min_dpt, dim]
        
        # choose optimum function
        if diffmap_min_dpt.mean() < diffmap_mean:
            opt = np.argmin
        else:
            opt = np.argmax
        # count opt cell
        min_dpt_cell[opt(diffmap_min_dpt)] += 1
    
    # root cell is cell with max vote
    return min_dpt[np.argmax(min_dpt_cell)]