# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:34:05 2020

@author: jlo
"""

import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import sklearn.cluster as skc

os.chdir(r"C:\Users\jlo\Documents\Summer20\HUANG\Data")
pp_donorA = "donorA/filtered_matrices_mex/hg19/write/donorA_pp.h5ad"
pp_donorB = "donorB/filtered_matrices_mex/hg19/write/donorB_pp.h5ad"
pp_donorA_sample = "donorA/filtered_matrices_mex/hg19/write/donorA_sample_pp.h5ad"
pp_donorB_sample = "donorB/filtered_matrices_mex/hg19/write/donorB_sample_pp.h5ad"
pp_donors_merged = "donors_merged/donors_merged_pp.h5ad"
pp_donors_merged_control = "donors_merged/donors_merged__control_pp.h5ad"

sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor = 'white')

DonorA = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\donorA\filtered_matrices_mex\hg19',var_names = 'gene_symbols', cache = True)
DonorB = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\donorB\filtered_matrices_mex\hg19',var_names = 'gene_symbols', cache = True)
DonorA_sample = sc.pp.subsample(DonorA, fraction = .1, copy = True)
DonorB_sample = sc.pp.subsample(DonorB, fraction = .1, copy = True)
Donors_merged_control = DonorA_sample.concatenate(DonorB_sample, join = 'outer')


#artificially change labels for donor B
oldvars = DonorB_sample.var_names
newvars = [str(i) for i in range(1,(len(DonorB_sample.var_names)+1))]
DonorB_sample.var_names = newvars
Donors_merged= DonorA_sample.concatenate(DonorB_sample, join = 'outer')

sc.pl.highest_expr_genes(DonorA, n_top=20,)
sc.pl.highest_expr_genes(DonorB, n_top=20,)
sc.pl.highest_expr_genes(DonorA_sample, n_top=20,)
sc.pl.highest_expr_genes(DonorB_sample, n_top=20,)
sc.pl.highest_expr_genes(Donors_merged, n_top = 20,)
sc.pl.highest_expr_genes(Donors_merged_control, n_top = 20,)
sc.pp.filter_cells(DonorA, min_genes = 200)
sc.pp.filter_cells(DonorB, min_genes = 200)
sc.pp.filter_cells(DonorA_sample, min_genes = 200)
sc.pp.filter_cells(DonorB_sample, min_genes = 200)
sc.pp.filter_cells(Donors_merged, min_genes = 200)
sc.pp.filter_cells(Donors_merged_control, min_genes = 200)
sc.pp.filter_genes(DonorA, min_cells=5)
sc.pp.filter_genes(DonorB, min_cells=5)
sc.pp.filter_genes(DonorA_sample, min_cells=5)
sc.pp.filter_genes(DonorB_sample, min_cells=5)
sc.pp.filter_genes(Donors_merged, min_cells = 5)
sc.pp.filter_genes(Donors_merged_control, min_cells = 5)
DonorA.var['mt'] = DonorA.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error
DonorB.var['mt'] = DonorB.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error

sc.pp.calculate_qc_metrics(DonorA, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)
sc.pp.calculate_qc_metrics(DonorB, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)
sc.pp.calculate_qc_metrics(Donors_merged, percent_top = None, log1p=False, inplace = True)
sc.pp.calculate_qc_metrics(Donors_merged_control, percent_top = None, log1p=False, inplace = True)
sc.pp.calculate_qc_metrics(DonorA_sample,  percent_top = None, log1p=False, inplace = True)
sc.pp.calculate_qc_metrics(DonorB_sample,  percent_top = None, log1p=False, inplace = True)


sc.pl.violin(DonorA, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .4, multi_panel = True)
sc.pl.violin(DonorB, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .4, multi_panel = True)
sc.pl.violin(DonorA_sample, ['n_genes_by_counts', 'total_counts'], jitter = .4, multi_panel = True)
sc.pl.violin(DonorB_sample, ['n_genes_by_counts', 'total_counts'], jitter = .4, multi_panel = True)
sc.pl.violin(Donors_merged, ['n_genes_by_counts', 'total_counts'], jitter = .4, multi_panel = True)
sc.pl.violin(Donors_merged_control, ['n_genes_by_counts', 'total_counts'], jitter = .4, multi_panel = True)


sc.pl.scatter(DonorA, x = 'total_counts', y = 'pct_counts_mt')
sc.pl.scatter(DonorB, x = 'total_counts', y = 'pct_counts_mt')#visualize - note that the mt percentage is on average higher for this dataset, and there appears to be a clear elbow starting at around 18
DonorA = DonorA[DonorA.obs.pct_counts_mt<5, :]
DonorB = DonorB[DonorB.obs.pct_counts_mt<6, :]
sc.pl.scatter(DonorA, x = 'total_counts', y = 'pct_counts_mt')#visualize after
sc.pl.scatter(DonorB, x = 'total_counts', y = 'pct_counts_mt')

sc.pl.scatter(DonorA, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorB, x = 'total_counts', y = 'n_genes_by_counts')#visualize
sc.pl.scatter(DonorA_sample, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorB_sample, x = 'total_counts', y = 'n_genes_by_counts')#visualize
sc.pl.scatter(Donors_merged, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(Donors_merged_control, x = 'total_counts', y = 'n_genes_by_counts')

DonorA = DonorA[DonorA.obs.n_genes_by_counts <2000, :]
DonorB = DonorB[DonorB.obs.n_genes_by_counts <2000, :]
DonorA_sample = DonorA_sample[DonorA_sample.obs.n_genes_by_counts <2000, :]
DonorB_sample = DonorB_sample[DonorB_sample.obs.n_genes_by_counts <2000, :]
Donors_merged = Donors_merged[Donors_merged.obs.n_genes_by_counts <2000, :]
Donors_merged_control = Donors_merged_control[Donors_merged_control.obs.n_genes_by_counts <2000, :]

sc.pl.scatter(DonorA, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorB, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorA_sample, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(DonorB_sample, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(Donors_merged, x = 'total_counts', y = 'n_genes_by_counts')
sc.pl.scatter(Donors_merged_control, x = 'total_counts', y = 'n_genes_by_counts')

sc.pp.normalize_total(DonorA)
sc.pp.normalize_total(DonorB)
sc.pp.normalize_total(DonorA_sample)
sc.pp.normalize_total(DonorB_sample)
sc.pp.normalize_total(Donors_merged)
sc.pp.normalize_total(Donors_merged_control)

sc.pp.log1p(DonorA)
sc.pp.log1p(DonorB)
sc.pp.log1p(DonorA_sample)
sc.pp.log1p(DonorB_sample)
sc.pp.log1p(Donors_merged)
sc.pp.log1p(Donors_merged_control)

sc.pp.highly_variable_genes(DonorA, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pp.highly_variable_genes(DonorB, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pp.highly_variable_genes(DonorA_sample, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pp.highly_variable_genes(DonorB_sample, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pp.highly_variable_genes(Donors_merged, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pp.highly_variable_genes(Donors_merged_control, min_mean = .0125, max_mean = 3, min_disp = .5)


sc.pl.highly_variable_genes(DonorA)
sc.pl.highly_variable_genes(DonorB)
sc.pl.highly_variable_genes(DonorA_sample)
sc.pl.highly_variable_genes(DonorB_sample)
sc.pl.highly_variable_genes(Donors_merged)
sc.pl.highly_variable_genes(Donors_merged_control)
DonorA = DonorA[:, DonorA.var.highly_variable]
DonorB = DonorB[:, DonorB.var.highly_variable]
DonorA_sample = DonorA_sample[:, DonorA_sample.var.highly_variable]
DonorB_sample = DonorB_sample[:, DonorB_sample.var.highly_variable]
Donors_merged = Donors_merged[:, Donors_merged.var.highly_variable]
Donors_merged_control = Donors_merged_control[:, Donors_merged_control.var.highly_variable]

sc.pp.regress_out(DonorA, ['total_counts', 'pct_counts_mt'])
sc.pp.regress_out(DonorB, ['total_counts', 'pct_counts_mt'])
sc.pp.regress_out(DonorA_sample, ['total_counts'])
sc.pp.regress_out(DonorB_sample, ['total_counts'])
sc.pp.regress_out(Donors_merged, ['total_counts'])
sc.pp.regress_out(Donors_merged_control, ['total_counts'])
##scale to unit variance
sc.pp.scale(DonorA, max_value = 10)
sc.pp.scale(DonorB, max_value = 10)
sc.pp.scale(DonorA_sample, max_value = 10)
sc.pp.scale(DonorB_sample, max_value = 10)
sc.pp.scale(Donors_merged, max_value = 10)
sc.pp.scale(Donors_merged_control, max_value = 10)


sc.tl.pca(DonorA, svd_solver='arpack')
sc.tl.pca(DonorB, svd_solver='arpack')
sc.tl.pca(DonorA_sample, svd_solver='arpack')
sc.tl.pca(DonorB_sample, svd_solver='arpack')
sc.tl.pca(Donors_merged, svd_solver = 'arpack')
sc.tl.pca(Donors_merged_control, svd_solver = 'arpack')
#save
DonorA.write(pp_donorA)
DonorB.write(pp_donorB)
DonorA_sample.write(pp_donorA_sample)
DonorB_sample.write(pp_donorB_sample)
Donors_merged.write(pp_donors_merged)
Donors_merged_control.write(pp_donors_merged_control)

#neighborhood graph
neighbors = 5
sc.pp.neighbors(DonorA, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(DonorB, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(DonorA_sample, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(DonorB_sample, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(Donors_merged, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(Donors_merged_control, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset


sc.tl.umap(DonorA)
sc.tl.umap(DonorB)
sc.tl.umap(Donors_merged)
sc.tl.umap(Donors_merged_control)
sc.tl.umap(DonorA_sample)
sc.tl.umap(DonorB_sample)
sc.pl.umap(DonorA, color = ['CST3'])
sc.pl.umap(DonorB, color = ['CST3'])
sc.pl.umap(Donors_merged_control, color = ['CST3'])

sc.pl.pca(DonorA)
sc.pl.pca(DonorB)
sc.pl.pca(DonorA_sample)
sc.pl.pca(DonorB_sample)
sc.pl.pca(Donors_merged, color = 'batch')
sc.pl.pca(Donors_merged_control, color = 'batch')

sc.tl.leiden(DonorA)
sc.tl.leiden(DonorB)
sc.tl.leiden(DonorA_sample)
sc.tl.leiden(DonorB_sample)
sc.tl.leiden(Donors_merged)
sc.tl.leiden(Donors_merged_control)

DonorA.uns['connectivities'] = DonorA.obsp['connectivities']
DonorB.uns['connectivities'] = DonorB.obsp['connectivities']
DonorA_sample.uns['connectivities'] = DonorA_sample.obsp['connectivities']
DonorB_sample.uns['connectivities'] = DonorB_sample.obsp['connectivities']
Donors_merged.uns['connectivities'] = Donors_merged.obsp['connectivities']
Donors_merged_control.uns['connectivities'] = Donors_merged_control.obsp['connectivities']

Donors_merged_asw = skm.silhouette_score(Donors_merged.X, Donors_merged.obs['batch'])
Donors_merged_control_asw = skm.silhouette_score(Donors_merged_control.X, Donors_merged_control.obs['batch'])
Donors_merged_umap_asw = skm.silhouette_score(Donors_merged.obsm['X_umap'], Donors_merged.obs['batch'])
Donors_merged_control_umap_asw = skm.silhouette_score(Donors_merged_control.obsm['X_umap'], Donors_merged_control.obs['batch'])

Donors_merged.obs['kmeans'] = skc.KMeans(n_clusters = 2).fit(Donors_merged.obsm['X_umap']).labels_
Donors_merged_control.obs['kmeans'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_control.obsm['X_umap']).labels_
Donors_merged_rand = skm.adjusted_rand_score(Donors_merged.obs['batch'], Donors_merged.obs['kmeans'])
Donors_merged_control_rand = skm.adjusted_rand_score(Donors_merged_control.obs['batch'], Donors_merged_control.obs['kmeans'])

Donors_merged_tsne = manifold.TSNE().fit_transform(Donors_merged.X)
Donors_merged_tsne = pd.DataFrame({'tsne-1': Donors_merged_tsne[:,0], 
                                 'tsne-2': Donors_merged_tsne[:,1], 
                                 'batch': Donors_merged.obs['batch']})
Donors_merged_control_tsne = manifold.TSNE().fit_transform(Donors_merged_control.X)
Donors_merged_control_tsne = pd.DataFrame({'tsne-1': Donors_merged_control_tsne[:,0], 
                                 'tsne-2': Donors_merged_control_tsne[:,1], 
                                 'batch': Donors_merged_control.obs['batch']})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_tsne,
                legend = "full").set_title("control, no correspondence")

plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_tsne,
                legend = "full").set_title("control, with correspondence")

reducer = umap.UMAP()
Donors_merged_umap = reducer.fit_transform(Donors_merged.X)
Donors_merged_umap = pd.DataFrame({'umap-1': Donors_merged_umap[:,0], 
                                 'umap-2': Donors_merged_umap[:,1], 
                                 'batch': Donors_merged.obs['batch']})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_umap,
                legend = "full").set_title("control, no correspondence")

reducer = umap.UMAP()
Donors_merged_control_umap = reducer.fit_transform(Donors_merged_control.X)
Donors_merged_control_umap = pd.DataFrame({'umap-1': Donors_merged_control_umap[:,0], 
                                 'umap-2': Donors_merged_control_umap[:,1], 
                                 'batch': Donors_merged_control.obs['batch']})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_umap,
                legend = "full").set_title("control, with correspondence")

np.savetxt("DonorA.csv", DonorA.X, delimiter = ",")
np.savetxt("DonorB.csv", DonorB.X, delimiter = ",")
np.savetxt("DonorA_sample.csv", DonorA_sample.X, delimiter = ",")
np.savetxt("DonorB_sample.csv", DonorB_sample.X, delimiter = ",")
np.savetxt("Donors_merged.csv", Donors_merged.X, delimiter = ",")
np.savetxt("Donors_merged_control.csv", Donors_merged_control.X, delimiter = ",")

np.savetxt("DonorA_umap.csv", DonorA.obsm['X_umap'], delimiter = ",")
np.savetxt("DonorB_umap.csv", DonorB.obsm['X_umap'], delimiter = ",")
np.savetxt("DonorA_sample_umap.csv", DonorA_sample.obsm['X_umap'], delimiter = ",")
np.savetxt("DonorB_sample_umap.csv", DonorB_sample.obsm['X_umap'], delimiter = ",")
np.savetxt("Donors_merged_umap.csv", Donors_merged.obsm['X_umap'], delimiter = ",")
np.savetxt("Donors_merged_control_umap.csv", Donors_merged_control.obsm['X_umap'], delimiter = ",")

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def visualize(data, data_integrated, title, datatype=None):

    dataset_num = len(data)

    styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 

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
        plt.title(title)
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
        plt.title(title)
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.legend()

    plt.tight_layout()
    plt.show()