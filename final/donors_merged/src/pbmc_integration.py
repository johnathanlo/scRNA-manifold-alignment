# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:29:03 2020

@author: jlo
"""

# =============================================================================
#  ____                                                                       |
# /_   |                                                                      |
#  |   |  /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
#  |   |  #############Data loading and pre-processing########################|
#  |___|  /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|   
# =============================================================================

import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import sklearn.cluster as skc
import csv
import seaborn as sns
import random

os.chdir(r"C:\Users\jlo\Documents\Summer20\HUANG\Data")
pp_donorA = "donors_merged/results/donorA_pp.h5ad"
pp_donorB = "donors_merged/results/donorB_pp.h5ad"
pp_donorA_sample = "donors_merged/results/donorA_sample_pp.h5ad"
pp_donorB_sample = "donors_merged/results/donorB_sample_pp.h5ad"
pp_donors_merged = "donors_merged/results/donors_merged_pp.h5ad" ###treat variables as totally without correspondence
pp_donors_merged_control = "donors_merged/results/donors_merged__control_pp.h5ad" ###allow normal correspondence between variables

sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor = 'white')

DonorA = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\donorA\filtered_matrices_mex\hg19',
                         var_names = 'gene_symbols', cache = True)
DonorB = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\donorB\filtered_matrices_mex\hg19',
                         var_names = 'gene_symbols', cache = True)
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

sc.pp.calculate_qc_metrics(DonorA, 
                           qc_vars = ['mt'], 
                           percent_top = None,
                           log1p=False, 
                           inplace = True)
sc.pp.calculate_qc_metrics(DonorB, 
                           qc_vars = ['mt'], 
                           percent_top = None, 
                           log1p=False, 
                           inplace = True)
sc.pp.calculate_qc_metrics(Donors_merged, 
                           percent_top = None, 
                           log1p=False, 
                           inplace = True)
sc.pp.calculate_qc_metrics(Donors_merged_control, 
                           percent_top = None, 
                           log1p=False, 
                           inplace = True)
sc.pp.calculate_qc_metrics(DonorA_sample,  
                           percent_top = None,
                           log1p=False, 
                           inplace = True)
sc.pp.calculate_qc_metrics(DonorB_sample,  
                           percent_top = None, 
                           log1p=False, 
                           inplace = True)


sc.pl.violin(DonorA, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
             jitter = .4, multi_panel = True)
sc.pl.violin(DonorB, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
             jitter = .4, multi_panel = True)
sc.pl.violin(DonorA_sample, ['n_genes_by_counts', 'total_counts'], 
             jitter = .4, multi_panel = True)
sc.pl.violin(DonorB_sample, ['n_genes_by_counts', 'total_counts'], 
             jitter = .4, multi_panel = True)
sc.pl.violin(Donors_merged, ['n_genes_by_counts', 'total_counts'], 
             jitter = .4, multi_panel = True)
sc.pl.violin(Donors_merged_control, ['n_genes_by_counts', 'total_counts'], 
             jitter = .4, multi_panel = True)


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


#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
##############################end preprocessing##########################################################
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#


# =============================================================================
#  _______                                                                    | 
# / ___   )                                                                   |
# \/   )  |                                                                   |
#     /   )                                                                   |
#   _/   /                                                                    |
#  /   _/     /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# (   (__/\   ###Controls, with correspondence and without correspondence#####|
# \_______/   /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# =============================================================================

##############################################################################
###########Cluster and evaluate for control with correspondence###############
##############################################################################

neighbors = 5
sc.pp.neighbors(Donors_merged_control, n_neighbors = neighbors, n_pcs = 50)
sc.tl.pca(Donors_merged_control, svd_solver = 'arpack')
sc.tl.leiden(Donors_merged_control)
sc.tl.umap(Donors_merged_control)
sc.tl.tsne(Donors_merged_control)
Donors_merged_control.obs['kmeans'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_control.obsm['X_umap']).labels_

####control ASW###
Donors_merged_control_asw = skm.silhouette_score(Donors_merged_control.X, 
                                                 Donors_merged_control.obs['batch'])
Donors_merged_control_umap_asw = skm.silhouette_score(Donors_merged_control.obsm['X_umap'],
                                                      Donors_merged_control.obs['batch'])

###control ARI###
Donors_merged_control_rand = skm.adjusted_rand_score(Donors_merged_control.obs['batch'], 
                                                     Donors_merged_control.obs['kmeans'])

###control PCA###
Donors_merged_control_graph_pca = pd.DataFrame({'pca-1': Donors_merged_control.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_control.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_control.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_graph_pca,
                legend = "full").set_title("control, no alignment, no pp")

###control tsne###
Donors_merged_control_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_control.obsm['X_tsne'][:,0],
                                       'tsne-2': Donors_merged_control.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_control.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_graph_tsne,
                legend = "full").set_title("control, no alignment, no pp")

###control umap###
Donors_merged_control_graph_umap = pd.DataFrame({'umap-1': Donors_merged_control.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_control.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_control.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_graph_umap,
                legend = "full").set_title("control, no alignment, no pp")

plt.show()

##############################################################################
###########Cluster and evaluate for control withuut correspondence############
##############################################################################

neighbors = 5
sc.pp.neighbors(Donors_merged, n_neighbors = neighbors, n_pcs = 50)
sc.tl.pca(Donors_merged, svd_solver = 'arpack')
sc.tl.leiden(Donors_merged)
sc.tl.umap(Donors_merged)
sc.tl.tsne(Donors_merged)
Donors_merged.obs['kmeans'] = skc.KMeans(n_clusters = 2).fit(Donors_merged.obsm['X_umap']).labels_

####control ASW###
Donors_merged_asw = skm.silhouette_score(Donors_merged.X, Donors_merged.obs['batch'])
Donors_merged_umap_asw = skm.silhouette_score(Donors_merged.obsm['X_umap'], Donors_merged.obs['batch'])

###control ARI###
Donors_merged_rand = skm.adjusted_rand_score(Donors_merged.obs['batch'], Donors_merged.obs['kmeans'])

###control PCA###
Donors_merged_graph_pca = pd.DataFrame({'pca-1': Donors_merged.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged.obsm['X_pca'][:,1],
                                 'batch': Donors_merged.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_graph_pca,
                legend = "full").set_title("control, no alignment, no pp")

###control tsne###
Donors_merged_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged.obsm['X_tsne'][:,0],
                                       'tsne-2': Donors_merged.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_graph_tsne,
                legend = "full").set_title("control, no alignment, no pp")

###control umap###
Donors_merged_graph_umap = pd.DataFrame({'umap-1': Donors_merged.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged.obsm['X_umap'][:,1],
                                       'batch': Donors_merged.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_graph_umap,
                legend = "full").set_title("control, no alignment, no pp")

plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
##############################end controls################################################################
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#

# =============================================================================
#  ______                                                                     |
# / ___  \                                                                    |
# \/   \  \                                                                   |
#    ___) /                                                                   |
#   (___ (                                                                    |
#       ) \   /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# /\___/  /   ############integrate with harmony##############################|
# \______/    /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# =============================================================================

###run harmony###
###credit: yyang43@mgh.harvard.edu,bli28@mgh.harvard.edu for pytorch
###implementation and Korsunsky et al, https://doi.org/10.1038/s41592-019-0619-0
###for original algorithm

from harmony import harmonize

Donors_merged_control_harmony = Donors_merged_control.copy()
Donors_merged_harmony = Donors_merged.copy()

merged_control_harmonized = harmonize(Donors_merged_control.obsm['X_pca'], 
                                      Donors_merged_control.obs, 
                                      batch_key = 'batch')
Donors_merged_control_harmony.obsm['X_harmony'] = merged_control_harmonized

merged_harmonized = harmonize(Donors_merged.obsm['X_pca'], 
                              Donors_merged.obs, 
                              batch_key = 'batch')
Donors_merged_harmony.obsm['X_harmony'] = merged_harmonized

##############################################################################
########evaluate results for data without correspondence######################
##############################################################################
neighbors = 5

sc.pp.neighbors(Donors_merged_harmony, n_neighbors = neighbors, use_rep = 'X_harmony', n_pcs = 50)
sc.tl.leiden(Donors_merged_harmony)
Donors_merged_harmony.obsm['X_pca'] = Donors_merged_harmony.obsm['X_harmony']
sc.tl.umap(Donors_merged_harmony)
sc.tl.tsne(Donors_merged_harmony, use_rep = 'X_harmony')
Donors_merged_harmony.obs['kmeans_harmony'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_harmony.obsm['X_harmony']).labels_

###ASW###
Donors_merged_harmony_asw = skm.silhouette_score(Donors_merged_harmony.obsm['X_harmony'], Donors_merged_harmony.obs['batch'])
Donors_merged_harmony_umap_asw = skm.silhouette_score(Donors_merged_harmony.obsm['X_umap'], Donors_merged_harmony.obs['batch'])

###ARI###
Donors_merged_harmony_rand = skm.adjusted_rand_score(Donors_merged_harmony.obs['batch'], Donors_merged_harmony.obs['kmeans_harmony'])

###PCA###
Donors_merged_harmony_graph_pca = pd.DataFrame({'pca-1': Donors_merged_harmony.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_harmony.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_harmony_graph_pca,
                legend = "full").set_title("harmony")

###tsne###
Donors_merged_harmony_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_harmony.obsm['X_tsne'][:,0],
                                       'tsne-2': Donors_merged_harmony.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_harmony_graph_tsne,
                legend = "full").set_title("harmony")

###umap###
Donors_merged_harmony_graph_umap = pd.DataFrame({'umap-1': Donors_merged_harmony.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_harmony.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_harmony_graph_umap,
                legend = "full").set_title("harmony")


##############################################################################
#####evaluate results for data with correspondence############################
##############################################################################
neighbors = 5

sc.pp.neighbors(Donors_merged_control_harmony, 
                use_rep = 'X_harmony', 
                n_neighbors = neighbors, 
                n_pcs = 50)
sc.tl.leiden(Donors_merged_control_harmony)
Donors_merged_control_harmony.obsm['X_pca'] = Donors_merged_control_harmony.obsm['X_harmony']
sc.tl.umap(Donors_merged_control_harmony)
sc.tl.tsne(Donors_merged_control_harmony, use_rep = 'X_harmony')
Donors_merged_control_harmony.obs['kmeans_harmony'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_control_harmony.obsm['X_harmony']).labels_

####ASW###
Donors_merged_control_harmony_asw = skm.silhouette_score(Donors_merged_control_harmony.obsm['X_harmony'], 
                                                         Donors_merged_control_harmony.obs['batch'])
Donors_merged_control_harmony_umap_asw = skm.silhouette_score(Donors_merged_control_harmony.obsm['X_umap'], 
                                                              Donors_merged_control_harmony.obs['batch'])

###ARI###
Donors_merged_control_harmony_rand = skm.adjusted_rand_score(Donors_merged_control_harmony.obs['batch'], 
                                                             Donors_merged_control_harmony.obs['kmeans_harmony'])

###PCA###
Donors_merged_control_harmony_graph_pca = pd.DataFrame({'pca-1': Donors_merged_control_harmony.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_control_harmony.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_control_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_harmony_graph_pca,
                legend = "full").set_title("harmony, no pp")

###tsne###
Donors_merged_control_harmony_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_control_harmony.obsm['X_tsne'][:,0],
                                       'tsne-2':Donors_merged_control_harmony.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_control_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_harmony_graph_tsne,
                legend = "full").set_title("harmony, no pp")

###umap###
Donors_merged_control_harmony_graph_umap = pd.DataFrame({'umap-1': Donors_merged_control_harmony.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_control_harmony.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_control_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_harmony_graph_umap,
                legend = "full").set_title("harmony, no pp")
plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
###################end harmony###########################################################################
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#

# =============================================================================
#     ___                                                                     |
#    /   )                                                                    |
#   / /) |                                                                    |
#  / (_) (_                                                                   |
# (____   _)                                                                  |
#      ) (     /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/|
#      | |     #############integrate with unioncom###########################|
#      (_)     /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/|
# =============================================================================
 
###run unioncom###
###credit: Cao Kai, https://github.com/caokai1073 ###

from unioncom import UnionCom 

Donors_merged_unioncom = Donors_merged.copy()
Donors_merged_control_unioncom = Donors_merged_control.copy()

data_control_A = Donors_merged_control.X[Donors_merged_control.obs.batch == '0'].copy()
data_control_B = Donors_merged_control.X[Donors_merged_control.obs.batch == '1'].copy()
data_A = Donors_merged.X[Donors_merged.obs.batch == '0'].copy()
data_B = Donors_merged.X[Donors_merged.obs.batch == '1'].copy()

merged_control_unioncom = UnionCom.fit_transform([data_control_A,data_control_B], datatype=None, epoch_pd=1000, epoch_DNN=100, epsilon=0.001, 
lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=666, delay=0, 
beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=50, test=False)

merged_unioncom = UnionCom.fit_transform([data_A,data_B], datatype=None, epoch_pd=1000, epoch_DNN=100, epsilon=0.001, 
lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=666, delay=0, 
beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=50, test=False)

merged_control_unioncom_final = np.vstack((merged_control_unioncom[0], merged_control_unioncom[1]))
Donors_merged_control_unioncom.obsm['X_unioncom'] = merged_control_unioncom_final
merged_unioncom_final = np.vstack((merged_unioncom[0], merged_unioncom[1]))
Donors_merged_unioncom.obsm['X_unioncom'] = merged_unioncom_final


##############################################################################
########evaluate results for data without correspondence######################
##############################################################################
neighbors = 5

sc.pp.neighbors(Donors_merged_unioncom, n_neighbors = neighbors, use_rep = 'X_unioncom', n_pcs = 50)
sc.tl.leiden(Donors_merged_unioncom)
Donors_merged_unioncom.obsm['X_pca'] = Donors_merged_unioncom.obsm['X_unioncom']
sc.tl.umap(Donors_merged_unioncom)
sc.tl.tsne(Donors_merged_unioncom, use_rep = 'X_unioncom')
Donors_merged_unioncom.obs['kmeans_unioncom'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_unioncom.obsm['X_unioncom']).labels_

###ASW###
Donors_merged_unioncom_asw = skm.silhouette_score(Donors_merged_unioncom.obsm['X_unioncom'], Donors_merged_unioncom.obs['batch'])
Donors_merged_unioncom_umap_asw = skm.silhouette_score(Donors_merged_unioncom.obsm['X_umap'], Donors_merged_unioncom.obs['batch'])

###ARI###
Donors_merged_unioncom_rand = skm.adjusted_rand_score(Donors_merged_unioncom.obs['batch'], Donors_merged_unioncom.obs['kmeans_unioncom'])

###PCA###
Donors_merged_unioncom_graph_pca = pd.DataFrame({'pca-1': Donors_merged_unioncom.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_unioncom.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_unioncom_graph_pca,
                legend = "full").set_title("unioncom")

###tsne###
Donors_merged_unioncom_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_unioncom.obsm['X_tsne'][:,0],
                                       'tsne-2': Donors_merged_unioncom.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_unioncom_graph_tsne,
                legend = "full").set_title("unioncom")

###umap###
Donors_merged_unioncom_graph_umap = pd.DataFrame({'umap-1': Donors_merged_unioncom.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_unioncom.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_unioncom_graph_umap,
                legend = "full").set_title("unioncom")

##############################################################################
#####evaluate results for data with correspondence############################
##############################################################################
neighbors = 5

sc.pp.neighbors(Donors_merged_control_unioncom, 
                use_rep = 'X_unioncom', 
                n_neighbors = neighbors, 
                n_pcs = 50)
sc.tl.leiden(Donors_merged_control_unioncom)
Donors_merged_control_unioncom.obsm['X_pca'] = Donors_merged_control_unioncom.obsm['X_unioncom']
sc.tl.umap(Donors_merged_control_unioncom)
sc.tl.tsne(Donors_merged_control_unioncom, use_rep = 'X_unioncom')
Donors_merged_control_unioncom.obs['kmeans_unioncom'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_control_unioncom.obsm['X_unioncom']).labels_

###ASW###
Donors_merged_control_unioncom_asw = skm.silhouette_score(Donors_merged_control_unioncom.obsm['X_unioncom'], 
                                                         Donors_merged_control_unioncom.obs['batch'])
Donors_merged_control_unioncom_umap_asw = skm.silhouette_score(Donors_merged_control_unioncom.obsm['X_umap'], 
                                                              Donors_merged_control_unioncom.obs['batch'])

###ARI###
Donors_merged_control_unioncom_rand = skm.adjusted_rand_score(Donors_merged_control_unioncom.obs['batch'], 
                                                             Donors_merged_control_unioncom.obs['kmeans_unioncom'])

###PCA###
Donors_merged_control_unioncom_graph_pca = pd.DataFrame({'pca-1': Donors_merged_control_unioncom.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_control_unioncom.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_control_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_unioncom_graph_pca,
                legend = "full").set_title("unioncom, correspondence")

###tsne###
Donors_merged_control_unioncom_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_control_unioncom.obsm['X_tsne'][:,0],
                                       'tsne-2':Donors_merged_control_unioncom.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_control_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_unioncom_graph_tsne,
                legend = "full").set_title("unioncom, correspondence")

###umap###
Donors_merged_control_unioncom_graph_umap = pd.DataFrame({'umap-1': Donors_merged_control_unioncom.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_control_unioncom.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_control_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_unioncom_graph_umap,
                legend = "full").set_title("unioncom, correspondence")
plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
###################end unioncom###########################################################################
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#


# =============================================================================
#  _______                                                                    |
# (  ____ \                                                                   |
# | (    \/                                                                   |
# | (____                                                                     |
# (_____ \                                                                    |
#       ) )   /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# /\____) )   ##########integrate with MMD-ResNet#############################|
# \______/    /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# =============================================================================
         
###run mmd-resnet###
###credit: Uri Shaham, uri.shaham@yale.edu###

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from matplotlib.ticker import NullFormatter
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors

from sklearn import decomposition
import argparse
from itertools import count

Donors_merged_mmd = Donors_merged.copy()
Donors_merged_control_mmd = Donors_merged.copy()

def compute_dist_mat(X, Y=None, device=torch.device("cpu")):
    """
    Computes nxm matrix of squared distances
    args:
        X: nxd tensor of data points
        Y: mxd tensor of data points (optional)
    """
    if Y is None:
        Y = X
       
    X = X.to(device=device)    
    Y = Y.to(device=device)  
    dtype = X.data.type()
    dist_mat = Variable(torch.Tensor(X.size()[0], Y.size()[0]).type(dtype)).to(device=device) 

    for i, row in enumerate(X.split(1)):
        r_v = row.expand_as(Y)
        sq_dist = torch.sum((r_v - Y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat

def nn_search(X, Y=None, k=10):
    """
    Computes nearest neighbors in Y for points in X
    args:
        X: nxd tensor of query points
        Y: mxd tensor of data points (optional)
        k: number of neighbors
    """
    if Y is None:
        Y = X
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Y)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids
    
def compute_scale(Dis, k=5):
    """
    Computes scale as the max distance to the k neighbor
    args:
        Dis: nxk' numpy array of distances (output of nn_search)
        k: number of neighbors
    """
    scale = np.median(Dis[:, k - 1])
    return scale

def compute_kernel_mat(D, scale, device=torch.device('cpu')):
     """
     Computes RBF kernal matrix
     args:
        D: nxn tenosr of squared distances
        scale: standard dev 
     """
     W = torch.exp(-D / (scale ** 2))

     return W 
 
def scatterHist(x1,x2, y1,y2, axis1='', axis2='', title='', name1='', name2='',
                plots_dir=''):
    nullfmt = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    fig = plt.figure(figsize=(8, 8))
       
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x1, x2, color = 'blue', s=3)
    axScatter.scatter(y1, y2, color = 'red', s=3) 


    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = np.max([np.max(np.fabs(x1)), np.max(np.fabs(x2))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x1, bins=bins, color = 'blue', density=True, stacked = True, histtype='step' )
    axHisty.hist(x2, bins=bins, orientation='horizontal', color = 'blue', density=True, stacked = True, histtype='step')
    axHistx.hist(y1, bins=bins, color = 'red', density=True, stacked = True, histtype='step')
    axHisty.hist(y2, bins=bins, orientation='horizontal', color = 'red', density=True, stacked = True, histtype='step')
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    axHistx.set_xticklabels([])
    axHistx.set_yticklabels([])
    axHisty.set_xticklabels([])
    axHisty.set_yticklabels([])
    axScatter.set_xlabel(axis1, fontsize=18)
    axScatter.set_ylabel(axis2, fontsize=18)
    
    axHistx.set_title(title, fontsize=18)
    axScatter.legend([name1, name2], fontsize=18)
    plt.show(block=False)
    if not plots_dir=='':
        fig.savefig(plots_dir+'/'+title+'.eps' ,format='eps')
# ==============================================================================
# =                                Input arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_blocks', 
                    type=int, 
                    default=2, 
                    help='Number of resNet blocks')
parser.add_argument('--file1', 
                    type=str, 
                    default="./data/sample1.csv",
                    help='path to file 1')
parser.add_argument('--file2', 
                    type=str, 
                    default="./data/sample2.csv",
                    help='path to file 2')
parser.add_argument("--scale_k",
                    type=int,
                    default=5,
                    help="Number of neighbors for determining the RBF scale")
parser.add_argument('--batch_size', 
                    type=int, 
                    default=128,
                    help='Batch size (default=128)')
parser.add_argument('--lr', 
                    type=float, 
                    default=1e-5,
                    help='learning_rate (default=1e-3)')
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-6,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=10,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-4,
                    help="l_2 weight penalty")
parser.add_argument("--epochs_wo_im",
                    type=int,
                    default=50,
                    help="Number of epochs without improvement before stopping")
parser.add_argument("--save_dir",
                    type=str,
                    default='./calibrated_data',
                    help="Directory for calibrated data")

args = parser.parse_args()

device = torch.device("cpu")

# ==============================================================================
# =                      Dataset - with correspondence                         =
# ==============================================================================

sample1 = Donors_merged_control.obsm['X_pca'][Donors_merged_control.obs.batch == '0'].copy()
sample2 = Donors_merged_control.obsm['X_pca'][Donors_merged_control.obs.batch == '1'].copy()

sample1_tensor = torch.Tensor(sample1.copy())
sample1_dataset = torch.utils.data.TensorDataset(sample1_tensor)

sample2_tensor = torch.Tensor(sample2.copy())
sample2_dataset = torch.utils.data.TensorDataset(sample2_tensor)

sample1_loader = torch.utils.data.DataLoader(sample1_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

sample2_loader = torch.utils.data.DataLoader(sample2_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

input_dim1 = sample1.shape[1]
input_dim2 = sample2.shape[1]
assert input_dim1 == input_dim2, "samples are of different dimensions"
input_dim = input_dim1

# ==============================================================================
# =                                    Model                                   =
# ==============================================================================

class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    
    def __init__(self, 
                 dim,
                 use_dropout=False):
        """Initialize the Resnet block"""
        
        super(ResnetBlock, self).__init__()
        self.block = self.build_resnet_block(dim,
                                             use_dropout)

    def build_resnet_block(self,
                           dim,
                           use_dropout=False):
    
        block = [torch.nn.Linear(dim, dim),
                 torch.nn.BatchNorm1d(dim),
                 torch.nn.PReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        block += [torch.nn.Linear(dim, dim),
                 torch.nn.BatchNorm1d(dim),
                 torch.nn.PReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        return nn.Sequential(*block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out
        

class Mmd_resnet(nn.Module):   
    def __init__(self, 
                 input_dim,
                 n_blocks,
                 use_dropout=False):
        super(Mmd_resnet, self).__init__()
        
        model = []
        for i in range(n_blocks):  # add resnet blocks layers
            model += [ResnetBlock(input_dim,
                                  use_dropout)]
            
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Forward function (with skip connections)"""
        out = input + self.model(input)  # add skip connection
        return out

        
mmd_resnet = Mmd_resnet(input_dim,
                        args.n_blocks)     
    
# ==============================================================================
# =                         Optimizer and Learning rate                        =
# ==============================================================================    

optim = torch.optim.SGD(mmd_resnet.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)    

def lambda_rule(epoch) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((epoch + 1) / args.decay_step_size))
    return np.power(args.lr_decay_factor, exponent)

scheduler = lr_scheduler.LambdaLR(optim, 
                                        lr_lambda=lambda_rule)   

def update_lr():
        """ Learning rate updater """
        
        scheduler.step()
        lr = optim.param_groups[0]['lr']
        if lr < args.min_lr:
            optim.param_groups[0]['lr'] = args.min_lr
            lr = optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 

# ==============================================================================
# =                              Training procedure                            =
# ==============================================================================    

def training_step(batch1, batch2):
    
    mmd_resnet.train(True)
    
    calibrated_batch2 = mmd_resnet(batch2)
    
    # Compute distance matrices
    D1 = compute_dist_mat(batch1, device=device)
    D2 = compute_dist_mat(calibrated_batch2, device=device)
    D12 = compute_dist_mat(batch1, calibrated_batch2)
    
    # Compute scale
    Dis, _ =nn_search(batch1, k=args.scale_k)
    scale = compute_scale(Dis, k=args.scale_k)
    
    # Compute kernel matrices
    K1 = compute_kernel_mat(D1, scale)   
    K2 = compute_kernel_mat(D2, scale) 
    K12 = compute_kernel_mat(D12, scale)
    
    # Loss function and backprop
    mmd = torch.mean(K1) - 2 * torch.mean(K12) + torch.mean(K2)    
    loss = torch.sqrt(mmd)  

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()
    
    
# ==============================================================================
# =                                     Main                                   =
# ==============================================================================    
     
    
best_loss = 100
eps = 1e-4
epoch_counter = 0
for epoch in count(1):
    batch_losses = []
    
    samp2_batches = enumerate(sample2_loader)
    for batch_idx, batch1 in enumerate(sample1_loader):
        try:
            _, batch2 = next(samp2_batches)
        except:
            samp2_batches = enumerate(sample2_loader)
            _, batch2 = next(samp2_batches)
            
        batch1 = batch1[0].to(device=device)
        batch2 = batch2[0].to(device=device)
        
        batch_loss = training_step(batch1, batch2)
        batch_losses.append(batch_loss)
    
    epoch_loss = np.mean(batch_losses)
    
    if epoch_loss < best_loss - eps:
        best_loss = epoch_loss
        epoch_counter = 0
    else:
        epoch_counter += 1
        
    print('Epoch {}, loss: {:.3f}, counter: {}'.format(epoch, 
                                          epoch_loss,
                                          epoch_counter)
    )
   
    update_lr()
    
    if epoch_counter == args.epochs_wo_im:
        break
print('Finished training')

# calibrate sample2 -> batch 1    

mmd_resnet.train(False)

calibrated_sample2 = []
for batch_idx, batch2 in enumerate(sample2_loader):
    batch2 = batch2[0].to(device=device)
    calibrated_batch = mmd_resnet(batch2)
    calibrated_sample2 += [calibrated_batch.detach().cpu().numpy()]
    
calibrated_sample2 = np.concatenate(calibrated_sample2)
           
# ==============================================================================
# =                         visualize calibration                              =
# ==============================================================================

# PCA
pca = decomposition.PCA()
pca.fit(sample1)
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)

# plot data before calibration
sample1_pca = pca.transform(sample1)
sample2_pca = pca.transform(sample2)
scatterHist(sample1_pca[:,pc1], 
               sample1_pca[:,pc2], 
               sample2_pca[:,pc1], 
               sample2_pca[:,pc2], 
               axis1, 
               axis2, 
               title="Data before calibration",
               name1='sample1', 
               name2='sample2')

# plot data after calibration
calibrated_sample2_pca = pca.transform(calibrated_sample2)
scatterHist(sample1_pca[:,pc1], 
               sample1_pca[:,pc2], 
               calibrated_sample2_pca[:,pc1], 
               calibrated_sample2_pca[:,pc2], 
               axis1, 
               axis2, 
               title="Data after calibration",
               name1='sample1', 
               name2='sample2')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
np.save(args.save_dir + '/sample1.csv', sample1)
np.save(args.save_dir + '/calibrated_sample2.csv', calibrated_sample2)

MMD_final = [sample1_pca, calibrated_sample2_pca]    
merged_control_mmd_final = np.vstack((MMD_final[0], MMD_final[1]))
Donors_merged_control_mmd.obsm['X_mmd'] = merged_control_mmd_final   

# ==============================================================================
# =                      Dataset - without correspondence                      =
# ==============================================================================

sample1 = Donors_merged.obsm['X_pca'][Donors_merged.obs.batch == '0'].copy()
sample2 = Donors_merged.obsm['X_pca'][Donors_merged.obs.batch == '1'].copy()

sample1_tensor = torch.Tensor(sample1.copy())
sample1_dataset = torch.utils.data.TensorDataset(sample1_tensor)

sample2_tensor = torch.Tensor(sample2.copy())
sample2_dataset = torch.utils.data.TensorDataset(sample2_tensor)

sample1_loader = torch.utils.data.DataLoader(sample1_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

sample2_loader = torch.utils.data.DataLoader(sample2_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

input_dim1 = sample1.shape[1]
input_dim2 = sample2.shape[1]
assert input_dim1 == input_dim2, "samples are of different dimensions"
input_dim = input_dim1

# ==============================================================================
# =                                    Model                                   =
# ==============================================================================

class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    
    def __init__(self, 
                 dim,
                 use_dropout=False):
        """Initialize the Resnet block"""
        
        super(ResnetBlock, self).__init__()
        self.block = self.build_resnet_block(dim,
                                             use_dropout)

    def build_resnet_block(self,
                           dim,
                           use_dropout=False):
    
        block = [torch.nn.Linear(dim, dim),
                 torch.nn.BatchNorm1d(dim),
                 torch.nn.PReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        block += [torch.nn.Linear(dim, dim),
                 torch.nn.BatchNorm1d(dim),
                 torch.nn.PReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
            
        return nn.Sequential(*block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out
        

class Mmd_resnet(nn.Module):   
    def __init__(self, 
                 input_dim,
                 n_blocks,
                 use_dropout=False):
        super(Mmd_resnet, self).__init__()
        
        model = []
        for i in range(n_blocks):  # add resnet blocks layers
            model += [ResnetBlock(input_dim,
                                  use_dropout)]
            
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Forward function (with skip connections)"""
        out = input + self.model(input)  # add skip connection
        return out

        
mmd_resnet = Mmd_resnet(input_dim,
                        args.n_blocks)     
    
# ==============================================================================
# =                         Optimizer and Learning rate                        =
# ==============================================================================    

optim = torch.optim.SGD(mmd_resnet.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)    

def lambda_rule(epoch) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((epoch + 1) / args.decay_step_size))
    return np.power(args.lr_decay_factor, exponent)

scheduler = lr_scheduler.LambdaLR(optim, 
                                        lr_lambda=lambda_rule)   

def update_lr():
        """ Learning rate updater """
        
        scheduler.step()
        lr = optim.param_groups[0]['lr']
        if lr < args.min_lr:
            optim.param_groups[0]['lr'] = args.min_lr
            lr = optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 

# ==============================================================================
# =                              Training procedure                            =
# ==============================================================================    

def training_step(batch1, batch2):
    
    mmd_resnet.train(True)
    
    calibrated_batch2 = mmd_resnet(batch2)
    
    # Compute distance matrices
    D1 = compute_dist_mat(batch1, device=device)
    D2 = compute_dist_mat(calibrated_batch2, device=device)
    D12 = compute_dist_mat(batch1, calibrated_batch2)
    
    # Compute scale
    Dis, _ =nn_search(batch1, k=args.scale_k)
    scale = compute_scale(Dis, k=args.scale_k)
    
    # Compute kernel matrices
    K1 = compute_kernel_mat(D1, scale)   
    K2 = compute_kernel_mat(D2, scale) 
    K12 = compute_kernel_mat(D12, scale)
    
    # Loss function and backprop
    mmd = torch.mean(K1) - 2 * torch.mean(K12) + torch.mean(K2)    
    loss = torch.sqrt(mmd)  

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()
    
    
# ==============================================================================
# =                                     Main                                   =
# ==============================================================================    
     
    
best_loss = 100
eps = 1e-4
epoch_counter = 0
for epoch in count(1):
    batch_losses = []
    
    samp2_batches = enumerate(sample2_loader)
    for batch_idx, batch1 in enumerate(sample1_loader):
        try:
            _, batch2 = next(samp2_batches)
        except:
            samp2_batches = enumerate(sample2_loader)
            _, batch2 = next(samp2_batches)
            
        batch1 = batch1[0].to(device=device)
        batch2 = batch2[0].to(device=device)
        
        batch_loss = training_step(batch1, batch2)
        batch_losses.append(batch_loss)
    
    epoch_loss = np.mean(batch_losses)
    
    if epoch_loss < best_loss - eps:
        best_loss = epoch_loss
        epoch_counter = 0
    else:
        epoch_counter += 1
        
    print('Epoch {}, loss: {:.3f}, counter: {}'.format(epoch, 
                                          epoch_loss,
                                          epoch_counter)
    )
   
    update_lr()
    
    if epoch_counter == args.epochs_wo_im:
        break
print('Finished training')

# calibrate sample2 -> batch 1    

mmd_resnet.train(False)

calibrated_sample2 = []
for batch_idx, batch2 in enumerate(sample2_loader):
    batch2 = batch2[0].to(device=device)
    calibrated_batch = mmd_resnet(batch2)
    calibrated_sample2 += [calibrated_batch.detach().cpu().numpy()]
    
calibrated_sample2 = np.concatenate(calibrated_sample2)
           
# ==============================================================================
# =                         visualize calibration                              =
# ==============================================================================

# PCA
pca = decomposition.PCA()
pca.fit(sample1)
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)

# plot data before calibration
sample1_pca = pca.transform(sample1)
sample2_pca = pca.transform(sample2)
scatterHist(sample1_pca[:,pc1], 
               sample1_pca[:,pc2], 
               sample2_pca[:,pc1], 
               sample2_pca[:,pc2], 
               axis1, 
               axis2, 
               title="Data before calibration",
               name1='sample1', 
               name2='sample2')

# plot data after calibration
calibrated_sample2_pca = pca.transform(calibrated_sample2)
scatterHist(sample1_pca[:,pc1], 
               sample1_pca[:,pc2], 
               calibrated_sample2_pca[:,pc1], 
               calibrated_sample2_pca[:,pc2], 
               axis1, 
               axis2, 
               title="Data after calibration",
               name1='sample1', 
               name2='sample2')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
np.save(args.save_dir + '/sample1.csv', sample1)
np.save(args.save_dir + '/calibrated_sample2.csv', calibrated_sample2)

MMD_final = [sample1_pca, calibrated_sample2_pca]    
merged_mmd_final = np.vstack((MMD_final[0], MMD_final[1]))
Donors_merged_mmd.obsm['X_mmd'] = merged_mmd_final     

##############################################################################
########evaluate results for data without correspondence######################
##############################################################################
neighbors = 5

sc.pp.neighbors(Donors_merged_mmd, n_neighbors = neighbors, use_rep = 'X_mmd', n_pcs = 50)
sc.tl.leiden(Donors_merged_mmd)
Donors_merged_mmd.obsm['X_pca'] = Donors_merged_mmd.obsm['X_mmd']
sc.tl.umap(Donors_merged_mmd)
sc.tl.tsne(Donors_merged_mmd, use_rep = 'X_mmd')
Donors_merged_mmd.obs['kmeans_mmd'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_mmd.obsm['X_mmd']).labels_

###ASW###
Donors_merged_mmd_asw = skm.silhouette_score(Donors_merged_mmd.obsm['X_mmd'], Donors_merged_mmd.obs['batch'])
Donors_merged_mmd_umap_asw = skm.silhouette_score(Donors_merged_mmd.obsm['X_umap'], Donors_merged_mmd.obs['batch'])

###ARI###
Donors_merged_mmd_rand = skm.adjusted_rand_score(Donors_merged_mmd.obs['batch'], Donors_merged_mmd.obs['kmeans_mmd'])

###PCA###
Donors_merged_mmd_graph_pca = pd.DataFrame({'pca-1': Donors_merged_mmd.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_mmd.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_mmd_graph_pca,
                legend = "full").set_title("mmd")

###tsne###
Donors_merged_mmd_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_mmd.obsm['X_tsne'][:,0],
                                       'tsne-2': Donors_merged_mmd.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_mmd_graph_tsne,
                legend = "full").set_title("mmd")

###umap###
Donors_merged_mmd_graph_umap = pd.DataFrame({'umap-1': Donors_merged_mmd.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_mmd.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_mmd_graph_umap,
                legend = "full").set_title("mmd")

##############################################################################
#####evaluate results for data with correspondence############################
##############################################################################
neighbors = 5

sc.pp.neighbors(Donors_merged_control_mmd, 
                use_rep = 'X_mmd', 
                n_neighbors = neighbors, 
                n_pcs = 50)
sc.tl.leiden(Donors_merged_control_mmd)
Donors_merged_control_mmd.obsm['X_pca'] = Donors_merged_control_mmd.obsm['X_mmd']
sc.tl.umap(Donors_merged_control_mmd)
sc.tl.tsne(Donors_merged_control_mmd, use_rep = 'X_mmd')
Donors_merged_control_mmd.obs['kmeans_mmd'] = skc.KMeans(n_clusters = 2).fit(Donors_merged_control_mmd.obsm['X_mmd']).labels_

####ASW###
Donors_merged_control_mmd_asw = skm.silhouette_score(Donors_merged_control_mmd.obsm['X_mmd'], 
                                                         Donors_merged_control_mmd.obs['batch'])
Donors_merged_control_mmd_umap_asw = skm.silhouette_score(Donors_merged_control_mmd.obsm['X_umap'], 
                                                              Donors_merged_control_mmd.obs['batch'])

###ARI###
Donors_merged_control_mmd_rand = skm.adjusted_rand_score(Donors_merged_control_mmd.obs['batch'], 
                                                             Donors_merged_control_mmd.obs['kmeans_mmd'])

###PCA###
Donors_merged_control_mmd_graph_pca = pd.DataFrame({'pca-1': Donors_merged_control_mmd.obsm['X_pca'][:,0],
                                 'pca-2': Donors_merged_control_mmd.obsm['X_pca'][:,1],
                                 'batch': Donors_merged_control_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_unioncom_graph_pca,
                legend = "full").set_title("mmd, correspondence")

###tsne###
Donors_merged_control_mmd_graph_tsne = pd.DataFrame({'tsne-1': Donors_merged_control_mmd.obsm['X_tsne'][:,0],
                                       'tsne-2':Donors_merged_control_mmd.obsm['X_tsne'][:,1],
                                       'batch': Donors_merged_control_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_mmd_graph_tsne,
                legend = "full").set_title("mmd, correspondence")

###umap###
Donors_merged_control_mmd_graph_umap = pd.DataFrame({'umap-1': Donors_merged_control_mmd.obsm['X_umap'][:,0],
                                       'umap-2': Donors_merged_control_mmd.obsm['X_umap'][:,1],
                                       'batch': Donors_merged_control_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = Donors_merged_control_mmd_graph_umap,
                legend = "full").set_title("mmd, correspondence")
plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#
###################end unioncom###########################################################################
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#


# =============================================================================
#   ______                                                                    |
#  / ____ \                                                                   |
# ( (    \/                                                                   |
# | (____                                                                     |
# |  ___ \                                                                    |
# | (   ) )   /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# ( (___) )   ##############Save and export data for analysis in R############|
#  \_____/    /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\|
# =============================================================================

###exports for kBET (note:kBET does not require clustered data but 
###LISI does)

#controls
np.savetxt("donors_merged/results/Donors_merged.csv", Donors_merged.X, delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control.csv", Donors_merged_control.X, delimiter = ",")

#harmony
np.savetxt("donors_merged/results/Donors_merged_harmony.csv", Donors_merged_harmony.obsm['X_harmony'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_harmony.csv", Donors_merged_control_harmony.obsm['X_harmony'], delimiter = ",")

#UnionCom
np.savetxt("donors_merged/results/Donors_merged_unioncom.csv", Donors_merged_unioncom.obsm['X_unioncom'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_unioncom.csv", Donors_merged_control_unioncom.obsm['X_unioncom'], delimiter = ",")

#MMD-ResNet
np.savetxt("donors_merged/results/Donors_merged_mmd.csv", Donors_merged_mmd.obsm['X_mmd'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_mmd.csv", Donors_merged_control_mmd.obsm['X_mmd'], delimiter = ",")


###exports for LISI

#control
np.savetxt("donors_merged/results/Donors_merged_umap.csv", Donors_merged.obsm['X_umap'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_umap.csv", Donors_merged_control.obsm['X_umap'], delimiter = ",")

#harmony
np.savetxt("donors_merged/results/Donors_merged_harmony_umap.csv", Donors_merged_harmony.obsm['X_umap'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_harmony_umap.csv", Donors_merged_control_harmony.obsm['X_umap'], delimiter = ",")

#unioncom
np.savetxt("donors_merged/results/Donors_merged_unioncom_umap.csv", Donors_merged_unioncom.obsm['X_umap'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_unioncom_umap.csv", Donors_merged_control_unioncom.obsm['X_umap'], delimiter = ",")

#mmd
np.savetxt("donors_merged/results/Donors_merged_mmd_umap.csv", Donors_merged_mmd.obsm['X_umap'], delimiter = ",")
np.savetxt("donors_merged/results/Donors_merged_control_mmd_umap.csv", Donors_merged_control_mmd.obsm['X_umap'], delimiter = ",")

###export table of ARI and ASW for each method

Donors_merged_ARI = np.asarray([Donors_merged_rand, 
                     Donors_merged_harmony_rand, 
                     Donors_merged_unioncom_rand, 
                     Donors_merged_mmd_rand])

Donors_merged_ASW = np.asarray([Donors_merged_asw,
                     Donors_merged_harmony_asw,
                     Donors_merged_unioncom_asw,
                     Donors_merged_mmd_asw])

Donors_merged_control_ARI = np.asarray([Donors_merged_control_rand,
                             Donors_merged_control_harmony_rand,
                             Donors_merged_control_unioncom_rand,
                             Donors_merged_control_mmd_rand])

Donors_merged_control_ASW = np.asarray([Donors_merged_control_asw,
                             Donors_merged_control_harmony_asw,
                             Donors_merged_control_unioncom_asw,
                             Donors_merged_control_mmd_asw])

Donors_ASW_ARI = pd.DataFrame({'ASW_no_correspondence': Donors_merged_ASW,
                                 'ARI_no_correspondence': Donors_merged_ARI,
                                 'ASW_correspondence': Donors_merged_control_ASW,
                                 'ARI_correspondence': Donors_merged_control_ARI}, 
                              index = ['control', 'harmony', 'unioncom','mmd'])

Donors_ASW_ARI.to_csv("donors_merged/results/Donors_ASW_ARI.csv")


# =============================================================================
# _________          _______    _______  _        ______                      |
# \__   __/|\     /|(  ____ \  (  ____ \( (    /|(  __  \                     |
#    ) (   | )   ( || (    \/  | (    \/|  \  ( || (  \  )                    | 
#    | |   | (___) || (__      | (__    |   \ | || |   ) |                    |
#    | |   |  ___  ||  __)     |  __)   | (\ \) || |   | |                    |  
#    | |   | (   ) || (        | (      | | \   || |   ) |                    |
#    | |   | )   ( || (____/\  | (____/\| )  \  || (__/  )                    |
#    )_(   |/     \|(_______/  (_______/|/    )_)(______/                     |
# =============================================================================
