# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:08:32 2020

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
dna_atac_pp = "dna_atac/results/dna_atac_pp.h5ad"

sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor = 'white')

atac_data = sc.read_csv(r"C:/Users/jlo/Documents/Summer20/HUANG/Data/dna_atac/data/dta_atac.csv", first_column_names = True).transpose()
rna_data = sc.read_csv(r"C:/Users/jlo/Documents/Summer20/HUANG/Data/dna_atac/data/dta_rna.csv", first_column_names = True).transpose()

merged_data_npp =  rna_data.concatenate(atac_data, join = 'inner')
merged_data =  rna_data.concatenate(atac_data, join = 'inner')
sc.pl.highest_expr_genes(merged_data, n_top = 20,)
##skip cell filtering: otherwise, indices of cells will get messed up##
sc.pp.filter_genes(merged_data, min_cells = 5)
##no MT genes##

#####QC#####
sc.pp.calculate_qc_metrics(merged_data, percent_top = None, log1p = False, inplace = True)
sc.pl.violin(merged_data, ['n_genes_by_counts','total_counts'], jitter = .2, multi_panel = True)
sc.pl.scatter(merged_data, x = 'total_counts', y = 'n_genes_by_counts')
sc.pp.normalize_total(merged_data)
sc.pp.log1p(merged_data)
sc.pp.highly_variable_genes(merged_data, min_mean = .0125, max_mean = 3, min_disp=.5)
merged_data = merged_data[:, merged_data.var.highly_variable]
sc.pp.regress_out(merged_data, ['total_counts'])
sc.pp.scale(merged_data, max_value = 10)


merged_data.write(dna_atac_pp)

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
###########Cluster and evaluate for control with preprocessing################
##############################################################################

neighbors = 5
sc.pp.neighbors(merged_data, n_neighbors = neighbors, n_pcs = 50)
sc.tl.leiden(merged_data)
sc.tl.pca(merged_data, svd_solver = 'arpack')
sc.tl.umap(merged_data)
sc.tl.tsne(merged_data)
merged_data.obs['kmeans'] = skc.KMeans(n_clusters = 2).fit(merged_data.obsm['X_umap']).labels_

####control ASW###
merged_data_asw = skm.silhouette_score(merged_data.X, merged_data.obs['batch'])
merged_data_umap_asw = skm.silhouette_score(merged_data.obsm['X_umap'], merged_data.obs['batch'])

###control ARI###
merged_data_rand = skm.adjusted_rand_score(merged_data.obs['batch'], merged_data.obs['kmeans'])

###control PCA###
merged_data_graph_pca = pd.DataFrame({'pca-1': merged_data.obsm['X_pca'][:,0],
                                 'pca-2': merged_data.obsm['X_pca'][:,1],
                                 'batch': merged_data.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_graph_pca,
                legend = "full").set_title("control, no alignment")

###control tsne###
merged_data_graph_tsne = pd.DataFrame({'tsne-1': merged_data.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data.obsm['X_tsne'][:,1],
                                       'batch': merged_data.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_graph_tsne,
                legend = "full").set_title("control, no alignment")

###control umap###
merged_data_graph_umap = pd.DataFrame({'umap-1': merged_data.obsm['X_umap'][:,0],
                                       'umap-2': merged_data.obsm['X_umap'][:,1],
                                       'batch': merged_data.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_graph_umap,
                legend = "full").set_title("control, no alignment")
plt.show()

########connect the same observations back to each other##########

###pca###
rna_data_pca = merged_data_graph_pca[0:len(rna_data)]
atac_data_pca = merged_data_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_pca.iloc[i,0], atac_data_pca.iloc[i,0]], 
             [rna_data_pca.iloc[i,1], atac_data_pca.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_pca.iloc[iterator,0], 
            rna_data_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_pca.iloc[iterator,0], 
            atac_data_pca.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, control - no alignment")
plt.xlabel("pca-1")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_tsne = merged_data_graph_tsne[0:len(rna_data)]
atac_data_tsne = merged_data_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_tsne.iloc[i,0], atac_data_tsne.iloc[i,0]], 
             [rna_data_tsne.iloc[i,1], atac_data_tsne.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_tsne.iloc[iterator,0],
            rna_data_tsne.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_tsne.iloc[iterator,0], 
            atac_data_tsne.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, control - no alignment")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_umap = merged_data_graph_umap[0:len(rna_data)]
atac_data_umap = merged_data_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_umap.iloc[i,0], atac_data_umap.iloc[i,0]], 
             [rna_data_umap.iloc[i,1], atac_data_umap.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_umap.iloc[iterator,0], 
            rna_data_umap.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_umap.iloc[iterator,0], 
            atac_data_umap.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, control - no alignment")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

plt.show()



##############################################################################
###########Cluster and evaluate for control without preprocessing#############
##############################################################################

neighbors = 5
sc.pp.neighbors(merged_data_npp, n_neighbors = neighbors, n_pcs = 50)
sc.tl.pca(merged_data_npp, svd_solver = 'arpack')
sc.tl.leiden(merged_data_npp)
sc.tl.umap(merged_data_npp)
sc.tl.tsne(merged_data_npp)
merged_data_npp.obs['kmeans'] = skc.KMeans(n_clusters = 2).fit(merged_data_npp.obsm['X_umap']).labels_

####control ASW###
merged_data_npp_asw = skm.silhouette_score(merged_data_npp.X, merged_data_npp.obs['batch'])
merged_data_npp_umap_asw = skm.silhouette_score(merged_data_npp.obsm['X_umap'], merged_data_npp.obs['batch'])

###control ARI###
merged_data_npp_rand = skm.adjusted_rand_score(merged_data_npp.obs['batch'], merged_data_npp.obs['kmeans'])

###control PCA###
merged_data_npp_graph_pca = pd.DataFrame({'pca-1': merged_data_npp.obsm['X_pca'][:,0],
                                 'pca-2': merged_data_npp.obsm['X_pca'][:,1],
                                 'batch': merged_data_npp.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_graph_pca,
                legend = "full").set_title("control, no alignment, no pp")

###control tsne###
merged_data_npp_graph_tsne = pd.DataFrame({'tsne-1': merged_data_npp.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data_npp.obsm['X_tsne'][:,1],
                                       'batch': merged_data_npp.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_graph_tsne,
                legend = "full").set_title("control, no alignment, no pp")

###control umap###
merged_data_npp_graph_umap = pd.DataFrame({'umap-1': merged_data_npp.obsm['X_umap'][:,0],
                                       'umap-2': merged_data_npp.obsm['X_umap'][:,1],
                                       'batch': merged_data_npp.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_graph_umap,
                legend = "full").set_title("control, no alignment, no pp")

plt.show()

########connect the same observations back to each other##########

###pca###
rna_data_npp_pca = merged_data_npp_graph_pca[0:len(rna_data)]
atac_data_npp_pca = merged_data_npp_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_pca.iloc[i,0], atac_data_npp_pca.iloc[i,0]], 
             [rna_data_npp_pca.iloc[i,1], atac_data_npp_pca.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_pca.iloc[iterator,0], 
            rna_data_npp_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_pca.iloc[iterator,0], 
            atac_data_npp_pca.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, control - no alignment, no pp")
plt.xlabel("pca-1")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_npp_tsne = merged_data_npp_graph_tsne[0:len(rna_data)]
atac_data_npp_tsne = merged_data_npp_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_tsne.iloc[i,0], atac_data_npp_tsne.iloc[i,0]], 
             [rna_data_npp_tsne.iloc[i,1], atac_data_npp_tsne.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_tsne.iloc[iterator,0], 
            rna_data_npp_tsne.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_tsne.iloc[iterator,0], 
            atac_data_npp_tsne.iloc[iterator,1],
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, control - no alignment, no pp")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_npp_umap = merged_data_npp_graph_umap[0:len(rna_data)]
atac_data_npp_umap = merged_data_npp_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_umap.iloc[i,0], atac_data_npp_umap.iloc[i,0]],
             [rna_data_npp_umap.iloc[i,1], atac_data_npp_umap.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_umap.iloc[iterator,0], 
            rna_data_npp_umap.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_umap.iloc[iterator,0], 
            atac_data_npp_umap.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, control - no alignment, no pp")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

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
merged_data_harmony = harmonize(merged_data.obsm['X_pca'], merged_data.obs, batch_key = 'batch')
merged_data_npp_harmony = harmonize(merged_data_npp.obsm['X_pca'], merged_data_npp.obs, batch_key = 'batch')

merged_data.obsm['X_harmony'] = merged_data_harmony
merged_data_npp.obsm['X_harmony'] = merged_data_npp_harmony


##############################################################################
########evaluate results for data with preprocessing##########################
##############################################################################
neighbors = 5
merged_data_harmony = merged_data.copy()

sc.pp.neighbors(merged_data_harmony, n_neighbors = neighbors, use_rep = 'X_harmony', n_pcs = 50)
sc.tl.leiden(merged_data_harmony)
merged_data_harmony.obsm['X_pca'] = merged_data_harmony.obsm['X_harmony']
sc.tl.umap(merged_data_harmony)
sc.tl.tsne(merged_data_harmony, use_rep = 'X_harmony')
merged_data_harmony.obs['kmeans_harmony'] = skc.KMeans(n_clusters = 2).fit(merged_data_harmony.obsm['X_harmony']).labels_

###control ASW###
merged_data_harmony_asw = skm.silhouette_score(merged_data_harmony.obsm['X_harmony'], merged_data_harmony.obs['batch'])
merged_data_harmony_umap_asw = skm.silhouette_score(merged_data_harmony.obsm['X_umap'], merged_data_harmony.obs['batch'])

###control ARI###
merged_data_harmony_rand = skm.adjusted_rand_score(merged_data_harmony.obs['batch'], merged_data_harmony.obs['kmeans_harmony'])

###control PCA###
merged_data_harmony_graph_pca = pd.DataFrame({'pca-1': merged_data_harmony.obsm['X_pca'][:,0],
                                 'pca-2': merged_data_harmony.obsm['X_pca'][:,1],
                                 'batch': merged_data_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_harmony_graph_pca,
                legend = "full").set_title("harmony")

###control tsne###
merged_data_harmony_graph_tsne = pd.DataFrame({'tsne-1': merged_data_harmony.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data_harmony.obsm['X_tsne'][:,1],
                                       'batch': merged_data_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_harmony_graph_tsne,
                legend = "full").set_title("harmony")

###control umap###
merged_data_harmony_graph_umap = pd.DataFrame({'umap-1': merged_data_harmony.obsm['X_umap'][:,0],
                                       'umap-2': merged_data_harmony.obsm['X_umap'][:,1],
                                       'batch': merged_data_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_harmony_graph_umap,
                legend = "full").set_title("harmony")

########connect the same observations back to each other##########

###pca###
rna_data_harmony_pca = merged_data_harmony_graph_pca[0:len(rna_data)]
atac_data_harmony_pca = merged_data_harmony_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_harmony_pca.iloc[i,0], atac_data_harmony_pca.iloc[i,0]], 
             [rna_data_harmony_pca.iloc[i,1], atac_data_harmony_pca.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_harmony_pca.iloc[iterator,0], 
            rna_data_harmony_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_harmony_pca.iloc[iterator,0], 
            atac_data_harmony_pca.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, harmony")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_harmony_tsne = merged_data_harmony_graph_tsne[0:len(rna_data)]
atac_data_harmony_tsne = merged_data_harmony_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_harmony_tsne.iloc[i,0], atac_data_harmony_tsne.iloc[i,0]], [rna_data_harmony_tsne.iloc[i,1], atac_data_harmony_tsne.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_harmony_tsne.iloc[iterator,0], rna_data_harmony_tsne.iloc[iterator,1], color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_harmony_tsne.iloc[iterator,0], atac_data_harmony_tsne.iloc[iterator,1], color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, harmony")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_harmony_umap = merged_data_harmony_graph_umap[0:len(rna_data)]
atac_data_harmony_umap = merged_data_harmony_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_harmony_umap.iloc[i,0], atac_data_harmony_umap.iloc[i,0]], [rna_data_harmony_umap.iloc[i,1], atac_data_harmony_umap.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_harmony_umap.iloc[iterator,0], rna_data_harmony_umap.iloc[iterator,1], color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_harmony_umap.iloc[iterator,0], atac_data_harmony_umap.iloc[iterator,1], color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, harmony")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

plt.show()

##############################################################################
#####evaluate results for data without preprocessing##########################
##############################################################################

neighbors = 5
merged_data_npp_harmony = merged_data_npp.copy()
sc.pp.neighbors(merged_data_npp_harmony, use_rep = 'X_harmony', n_neighbors = neighbors, n_pcs = 50)
sc.tl.leiden(merged_data_npp_harmony)
merged_data_npp_harmony.obsm['X_pca'] = merged_data_npp_harmony.obsm['X_harmony']
sc.tl.umap(merged_data_npp_harmony)
sc.tl.tsne(merged_data_npp_harmony, use_rep = 'X_harmony')
merged_data_npp_harmony.obs['kmeans_harmony'] = skc.KMeans(n_clusters = 2).fit(merged_data_npp_harmony.obsm['X_harmony']).labels_

####control ASW###
merged_data_npp_harmony_asw = skm.silhouette_score(merged_data_npp_harmony.obsm['X_harmony'], merged_data_npp_harmony.obs['batch'])
merged_data_npp_harmony_umap_asw = skm.silhouette_score(merged_data_npp_harmony.obsm['X_umap'], merged_data_npp_harmony.obs['batch'])

###control ARI###
merged_data_npp_harmony_rand = skm.adjusted_rand_score(merged_data_npp_harmony.obs['batch'], merged_data_npp_harmony.obs['kmeans_harmony'])

###control PCA###
merged_data_npp_harmony_graph_pca = pd.DataFrame({'pca-1': merged_data_npp_harmony.obsm['X_pca'][:,0],
                                 'pca-2': merged_data_npp_harmony.obsm['X_pca'][:,1],
                                 'batch': merged_data_npp_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_harmony_graph_pca,
                legend = "full").set_title("harmony, no pp")

###control tsne###
merged_data_npp_harmony_graph_tsne = pd.DataFrame({'tsne-1': merged_data_npp_harmony.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data_npp_harmony.obsm['X_tsne'][:,1],
                                       'batch': merged_data_npp_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_harmony_graph_tsne,
                legend = "full").set_title("harmony, no pp")

###control umap###
merged_data_npp_harmony_graph_umap = pd.DataFrame({'umap-1': merged_data_npp_harmony.obsm['X_umap'][:,0],
                                       'umap-2': merged_data_npp_harmony.obsm['X_umap'][:,1],
                                       'batch': merged_data_npp_harmony.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_harmony_graph_umap,
                legend = "full").set_title("harmony, no pp")
plt.show()


########connect the same observations back to each other##########

###pca###
rna_data_npp_harmony_pca = merged_data_npp_harmony_graph_pca[0:len(rna_data)]
atac_data_npp_harmony_pca = merged_data_npp_harmony_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_harmony_pca.iloc[i,0], atac_data_npp_harmony_pca.iloc[i,0]], [rna_data_npp_harmony_pca.iloc[i,1], atac_data_npp_harmony_pca.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_harmony_pca.iloc[iterator,0], 
            rna_data_npp_harmony_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_harmony_pca.iloc[iterator,0],
            atac_data_npp_harmony_pca.iloc[iterator,1],
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, harmony no pp")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_npp_harmony_tsne = merged_data_npp_harmony_graph_tsne[0:len(rna_data)]
atac_data_npp_harmony_tsne = merged_data_npp_harmony_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_harmony_tsne.iloc[i,0], atac_data_npp_harmony_tsne.iloc[i,0]], 
             [rna_data_npp_harmony_tsne.iloc[i,1], atac_data_npp_harmony_tsne.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_harmony_tsne.iloc[iterator,0], 
            rna_data_npp_harmony_tsne.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_harmony_tsne.iloc[iterator,0], 
            atac_data_npp_harmony_tsne.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, harmony no pp")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_npp_harmony_umap = merged_data_npp_harmony_graph_umap[0:len(rna_data)]
atac_data_npp_harmony_umap = merged_data_npp_harmony_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_harmony_umap.iloc[i,0], atac_data_npp_harmony_umap.iloc[i,0]], 
             [rna_data_npp_harmony_umap.iloc[i,1], atac_data_npp_harmony_umap.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_harmony_umap.iloc[iterator,0], 
            rna_data_npp_harmony_umap.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_harmony_umap.iloc[iterator,0], 
            atac_data_npp_harmony_umap.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, harmony no pp")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

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

data_rna = merged_data.X[merged_data.obs.batch == '0'].copy()
data_atac = merged_data.X[merged_data.obs.batch == '1'].copy()
data_npp_rna = merged_data_npp.X[merged_data_npp.obs.batch == '0'].copy()
data_npp_atac = merged_data_npp.X[merged_data_npp.obs.batch == '1'].copy()

device = torch.device("cpu")

merged_unioncom = UnionCom.fit_transform([data_rna, data_atac], datatype=None, epoch_pd=1000, epoch_DNN=100, epsilon=0.001, 
lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=666, delay=0, 
beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=50, test=False)

# =============================================================================
# merged_npp_unioncom = UnionCom.fit_transform([data_npp_rna, data_npp_atac], datatype=None, epoch_pd=1000, epoch_DNN=100, epsilon=0.001, 
# lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=666, delay=0, 
# beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=50, test=False)
# =============================================================================doesn't work, not enough memory

merged_unioncom_final = np.vstack((merged_unioncom[0], merged_unioncom[1]))
merged_data.obsm['X_unioncom'] = merged_unioncom_final

##############################################################################
########evaluate results for data with preprocessing##########################
##############################################################################
neighbors = 5
merged_data_unioncom = merged_data.copy()

sc.pp.neighbors(merged_data_unioncom, n_neighbors = neighbors, use_rep = 'X_unioncom', n_pcs = 50)
sc.tl.leiden(merged_data_unioncom)
merged_data_unioncom.obsm['X_pca'] = merged_data_unioncom.obsm['X_unioncom']
sc.tl.umap(merged_data_unioncom)
sc.tl.tsne(merged_data_unioncom, use_rep = 'X_unioncom')
merged_data_unioncom.obs['kmeans_unioncom'] = skc.KMeans(n_clusters = 2).fit(merged_data_unioncom.obsm['X_unioncom']).labels_

###control ASW###
merged_data_unioncom_asw = skm.silhouette_score(merged_data_unioncom.obsm['X_unioncom'], merged_data_unioncom.obs['batch'])
merged_data_unioncom_umap_asw = skm.silhouette_score(merged_data_unioncom.obsm['X_umap'], merged_data_unioncom.obs['batch'])

###control ARI###
merged_data_unioncom_rand = skm.adjusted_rand_score(merged_data_unioncom.obs['batch'], merged_data_unioncom.obs['kmeans_unioncom'])

###control PCA###
merged_data_unioncom_graph_pca = pd.DataFrame({'pca-1': merged_data_unioncom.obsm['X_pca'][:,0],
                                 'pca-2': merged_data_unioncom.obsm['X_pca'][:,1],
                                 'batch': merged_data_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_unioncom_graph_pca,
                legend = "full").set_title("unioncom")

###control tsne###
merged_data_unioncom_graph_tsne = pd.DataFrame({'tsne-1': merged_data_unioncom.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data_unioncom.obsm['X_tsne'][:,1],
                                       'batch': merged_data_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_unioncom_graph_tsne,
                legend = "full").set_title("unioncom")

###control umap###
merged_data_unioncom_graph_umap = pd.DataFrame({'umap-1': merged_data_unioncom.obsm['X_umap'][:,0],
                                       'umap-2': merged_data_unioncom.obsm['X_umap'][:,1],
                                       'batch': merged_data_unioncom.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_unioncom_graph_umap,
                legend = "full").set_title("unioncom")

########connect the same observations back to each other##########

###pca###
rna_data_unioncom_pca = merged_data_unioncom_graph_pca[0:len(rna_data)]
atac_data_unioncom_pca = merged_data_unioncom_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_unioncom_pca.iloc[i,0], atac_data_unioncom_pca.iloc[i,0]], 
             [rna_data_unioncom_pca.iloc[i,1], atac_data_unioncom_pca.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_unioncom_pca.iloc[iterator,0], 
            rna_data_unioncom_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_unioncom_pca.iloc[iterator,0], 
            atac_data_unioncom_pca.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, unioncom")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_unioncom_tsne = merged_data_unioncom_graph_tsne[0:len(rna_data)]
atac_data_unioncom_tsne = merged_data_unioncom_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_unioncom_tsne.iloc[i,0], atac_data_unioncom_tsne.iloc[i,0]], 
             [rna_data_unioncom_tsne.iloc[i,1], atac_data_unioncom_tsne.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_unioncom_tsne.iloc[iterator,0], 
            rna_data_unioncom_tsne.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_unioncom_tsne.iloc[iterator,0], 
            atac_data_unioncom_tsne.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, unioncom")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_unioncom_umap = merged_data_unioncom_graph_umap[0:len(rna_data)]
atac_data_unioncom_umap = merged_data_unioncom_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_unioncom_umap.iloc[i,0], atac_data_unioncom_umap.iloc[i,0]], 
             [rna_data_unioncom_umap.iloc[i,1], atac_data_unioncom_umap.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_unioncom_umap.iloc[iterator,0], 
            rna_data_unioncom_umap.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_unioncom_umap.iloc[iterator,0], 
            atac_data_unioncom_umap.iloc[iterator,1],
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, unioncom")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

plt.show()

# =============================================================================
# ########################################################################
# #####clustering before alignment using unprocessed data#################
# ########################################################################
# 
# neighbors = 5
# merged_data_npp_harmony = merged_data_npp
# sc.pp.neighbors(merged_data_npp_harmony, use_rep = 'X_harmony', n_neighbors = neighbors, n_pcs = 50)
# sc.tl.leiden(merged_data_npp)
# merged_data_npp_harmony.obsm['X_pca'] = merged_data_npp_harmony.obsm['X_harmony']
# sc.tl.umap(merged_data_npp_harmony)
# sc.tl.tsne(merged_data_npp_harmony, use_rep = 'X_harmony')
# merged_data_npp_harmony.obs['kmeans_harmony'] = skc.KMeans(n_clusters = 2).fit(merged_data_npp_harmony.obsm['X_harmony']).labels_
# 
# ####control ASW###
# merged_data_npp_harmony_asw = skm.silhouette_score(merged_data_npp_harmony.obsm['X_harmony'], merged_data_npp_harmony.obs['batch'])
# merged_data_npp_harmony_umap_asw = skm.silhouette_score(merged_data_npp_harmony.obsm['X_umap'], merged_data_npp_harmony.obs['batch'])
# 
# ###control ARI###
# merged_data_npp_harmony_rand = skm.adjusted_rand_score(merged_data_npp_harmony.obs['batch'], merged_data_npp_harmony.obs['kmeans_harmony'])
# 
# ###control PCA###
# merged_data_npp_harmony_graph_pca = pd.DataFrame({'pca-1': merged_data_npp_harmony.obsm['X_pca'][:,0],
#                                  'pca-2': merged_data_npp_harmony.obsm['X_pca'][:,1],
#                                  'batch': merged_data_npp_harmony.obs.batch})
# plt.figure(figsize = (16,10))
# sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
#                 palette = sns.color_palette("hls", 2),
#                 data = merged_data_npp_harmony_graph_pca,
#                 legend = "full").set_title("harmony, no pp")
# 
# ###control tsne###
# merged_data_npp_harmony_graph_tsne = pd.DataFrame({'tsne-1': merged_data_npp_harmony.obsm['X_tsne'][:,0],
#                                        'tsne-2': merged_data_npp_harmony.obsm['X_tsne'][:,1],
#                                        'batch': merged_data_npp_harmony.obs.batch})
# plt.figure(figsize = (16,10))
# sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
#                 palette = sns.color_palette("hls", 2),
#                 data = merged_data_npp_harmony_graph_tsne,
#                 legend = "full").set_title("harmony, no pp")
# 
# ###control umap###
# merged_data_npp_harmony_graph_umap = pd.DataFrame({'umap-1': merged_data_npp_harmony.obsm['X_umap'][:,0],
#                                        'umap-2': merged_data_npp_harmony.obsm['X_umap'][:,1],
#                                        'batch': merged_data_npp_harmony.obs.batch})
# plt.figure(figsize = (16,10))
# sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
#                 palette = sns.color_palette("hls", 2),
#                 data = merged_data_npp_harmony_graph_umap,
#                 legend = "full").set_title("harmony, no pp")
# 
# 
# ########connect the same observations back to each other##########
# 
# ###pca###
# rna_data_npp_harmony_pca = merged_data_npp_harmony_graph_pca[0:len(rna_data)]
# atac_data_npp_harmony_pca = merged_data_npp_harmony_graph_pca[len(atac_data):]
# iterator = random.sample(range(0,len(rna_data),1), 20)
# 
# for i in iterator:
#     plt.plot([rna_data_npp_harmony_pca.iloc[i,0], atac_data_npp_harmony_pca.iloc[i,0]], [rna_data_npp_harmony_pca.iloc[i,1], atac_data_npp_harmony_pca.iloc[i,1]],'r-' , linewidth = .2)
#     
# plt.scatter(rna_data_npp_harmony_pca.iloc[iterator,0], 
#             rna_data_npp_harmony_pca.iloc[iterator,1], 
#             color = 'b', s = 3, label = "RNA-seq")
# plt.scatter(atac_data_npp_harmony_pca.iloc[iterator,0],
#             atac_data_npp_harmony_pca.iloc[iterator,1],
#             color = 'g', s = 3, label = "ATAC-seq")
# plt.title("PCA, harmony no pp")
# plt.ylabel("pca-2")
# plt.legend(loc = "best")
# 
# plt.show()
# 
# ###tsne###
# rna_data_npp_harmony_tsne = merged_data_npp_harmony_graph_tsne[0:len(rna_data)]
# atac_data_npp_harmony_tsne = merged_data_npp_harmony_graph_tsne[len(atac_data):]
# iterator = random.sample(range(0,len(rna_data),1), 20)
# 
# for i in iterator:
#     plt.plot([rna_data_npp_harmony_tsne.iloc[i,0], atac_data_npp_harmony_tsne.iloc[i,0]], 
#              [rna_data_npp_harmony_tsne.iloc[i,1], atac_data_npp_harmony_tsne.iloc[i,1]],
#              'r-' , linewidth = .2)
#     
# plt.scatter(rna_data_npp_harmony_tsne.iloc[iterator,0], 
#             rna_data_npp_harmony_tsne.iloc[iterator,1], 
#             color = 'b', s = 3, label = "RNA-seq")
# plt.scatter(atac_data_npp_harmony_tsne.iloc[iterator,0], 
#             atac_data_npp_harmony_tsne.iloc[iterator,1], 
#             color = 'g', s = 3, label = "ATAC-seq")
# plt.title("t-SNE, harmony no pp")
# plt.xlabel("tsne-1")
# plt.ylabel("tsne-2")
# plt.legend(loc = "best")
# 
# plt.show()
# 
# ###umap###
# rna_data_npp_harmony_umap = merged_data_npp_harmony_graph_umap[0:len(rna_data)]
# atac_data_npp_harmony_umap = merged_data_npp_harmony_graph_umap[len(atac_data):]
# iterator = random.sample(range(0,len(rna_data),1), 20)
# 
# for i in iterator:
#     plt.plot([rna_data_npp_harmony_umap.iloc[i,0], atac_data_npp_harmony_umap.iloc[i,0]], 
#              [rna_data_npp_harmony_umap.iloc[i,1], atac_data_npp_harmony_umap.iloc[i,1]],'r-' , linewidth = .2)
#     
# plt.scatter(rna_data_npp_harmony_umap.iloc[iterator,0], 
#             rna_data_npp_harmony_umap.iloc[iterator,1], 
#             color = 'b', s = 3, label = "RNA-seq")
# plt.scatter(atac_data_npp_harmony_umap.iloc[iterator,0], 
#             atac_data_npp_harmony_umap.iloc[iterator,1], 
#             color = 'g', s = 3, label = "ATAC-seq")
# plt.title("UMAP, harmony no pp")
# plt.xlabel("umap-1")
# plt.ylabel("umap-2")
# plt.legend(loc = "best")
# 
# plt.show()
# ============================================================================= no npp data

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

from sklearn import decomposition
import argparse
from itertools import count

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from matplotlib.ticker import NullFormatter
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors

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
                    default=256,
                    help='Batch size (default=128)')
parser.add_argument('--lr', 
                    type=float, 
                    default=1e-3,
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
                    default=100,
                    help="Number of epochs without improvement before stopping")
parser.add_argument("--save_dir",
                    type=str,
                    default='./calibrated_data',
                    help="Directory for calibrated data")

args = parser.parse_args()

device = torch.device("cpu")

# ==============================================================================
# =                                   Dataset                                  =
# ==============================================================================

sample1 = merged_data.obsm['X_pca'][merged_data.obs.batch == '0'].copy()
sample2 = merged_data.obsm['X_pca'][merged_data.obs.batch == '1'].copy()


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
# =                                    Model - processed                       =
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
# =                                     Main - processed data                  =
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
merged_data.obsm['X_mmd'] = merged_mmd_final

# ==============================================================================
# =                                   Dataset - npp                            =
# ==============================================================================


sample1_npp = merged_data_npp.obsm['X_pca'][merged_data_npp.obs.batch == '0'].copy()
sample2_npp = merged_data_npp.obsm['X_pca'][merged_data_npp.obs.batch == '1'].copy()

sample1_tensor = torch.Tensor(sample1_npp.copy())
sample1_dataset = torch.utils.data.TensorDataset(sample1_tensor)

sample2_tensor = torch.Tensor(sample2_npp.copy())
sample2_dataset = torch.utils.data.TensorDataset(sample2_tensor)

sample1_npp_loader = torch.utils.data.DataLoader(sample1_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

sample2_npp_loader = torch.utils.data.DataLoader(sample2_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

input_dim1 = sample1_npp.shape[1]
input_dim2 = sample2_npp.shape[1]

assert input_dim1 == input_dim2, "samples are of different dimensions"
input_dim = input_dim1

# ==============================================================================
# =                                    Model                      =
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
# =                                     Main - npp data                        =
# ==============================================================================    
     
    
best_loss = 100
eps = 1e-4
epoch_counter = 0
for epoch in count(1):
    batch_losses = []
    
    samp2_batches = enumerate(sample2_npp_loader)
    for batch_idx, batch1 in enumerate(sample1_npp_loader):
        try:
            _, batch2 = next(samp2_batches)
        except:
            samp2_batches = enumerate(sample2_npp_loader)
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

calibrated_sample2_npp = []
for batch_idx, batch2 in enumerate(sample2_npp_loader):
    batch2 = batch2[0].to(device=device)
    calibrated_batch = mmd_resnet(batch2)
    calibrated_sample2_npp += [calibrated_batch.detach().cpu().numpy()]
    
calibrated_sample2_npp = np.concatenate(calibrated_sample2_npp)
           
# ==============================================================================
# =                         visualize calibration - npp                        =
# ==============================================================================

# PCA
pca = decomposition.PCA()
pca.fit(sample1_npp)
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)

# plot data before calibration
sample1_npp_pca = pca.transform(sample1_npp)
sample2_npp_pca = pca.transform(sample2_npp)
scatterHist(sample1_npp_pca[:,pc1], 
               sample1_npp_pca[:,pc2], 
               sample2_npp_pca[:,pc1], 
               sample2_npp_pca[:,pc2], 
               axis1, 
               axis2, 
               title="Data before calibration",
               name1='sample1', 
               name2='sample2')

# plot data after calibration
calibrated_sample2_npp_pca = pca.transform(calibrated_sample2_npp)
scatterHist(sample1_npp_pca[:,pc1], 
               sample1_npp_pca[:,pc2], 
               calibrated_sample2_npp_pca[:,pc1], 
               calibrated_sample2_npp_pca[:,pc2], 
               axis1, 
               axis2, 
               title="Data after calibration",
               name1='sample1', 
               name2='sample2')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
np.save(args.save_dir + '/sample1_npp.csv', sample1_npp)
np.save(args.save_dir + '/calibrated_sample2_npp.csv', calibrated_sample2_npp)

MMD_final_npp = [sample1_npp_pca, calibrated_sample2_npp_pca] 

merged_mmd_npp_final = np.vstack((MMD_final_npp[0], MMD_final_npp[1]))
merged_data_npp.obsm['X_mmd'] = merged_mmd_npp_final

##############################################################################
########evaluate results for data with preprocessing##########################
##############################################################################
neighbors = 5
merged_data_mmd = merged_data.copy()

sc.pp.neighbors(merged_data_mmd , n_neighbors = neighbors, use_rep = 'X_mmd', n_pcs = 50)
sc.tl.leiden(merged_data_mmd )
merged_data_mmd.obsm['X_pca'] = merged_data_mmd.obsm['X_mmd']
sc.tl.umap(merged_data_mmd)
sc.tl.tsne(merged_data_mmd, use_rep = 'X_mmd')
merged_data_mmd.obs['kmeans_mmd'] = skc.KMeans(n_clusters = 2).fit(merged_data_mmd.obsm['X_mmd']).labels_

###mmd ASW###
merged_data_mmd_asw = skm.silhouette_score(merged_data_mmd.obsm['X_mmd'], merged_data_mmd.obs['batch'])
merged_data_mmd_umap_asw = skm.silhouette_score(merged_data_mmd.obsm['X_umap'], merged_data_mmd.obs['batch'])

###mmd ARI###
merged_data_mmd_rand = skm.adjusted_rand_score(merged_data_mmd.obs['batch'], merged_data_mmd.obs['kmeans_mmd'])

###mmd PCA###
merged_data_mmd_graph_pca = pd.DataFrame({'pca-1': merged_data_mmd.obsm['X_pca'][:,0],
                                 'pca-2': merged_data_mmd.obsm['X_pca'][:,1],
                                 'batch': merged_data_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_mmd_graph_pca,
                legend = "full").set_title("mmd")

###control tsne###
merged_data_mmd_graph_tsne = pd.DataFrame({'tsne-1': merged_data_mmd.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data_mmd.obsm['X_tsne'][:,1],
                                       'batch': merged_data_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_mmd_graph_tsne,
                legend = "full").set_title("mmd")

###control umap###
merged_data_mmd_graph_umap = pd.DataFrame({'umap-1': merged_data_mmd.obsm['X_umap'][:,0],
                                       'umap-2': merged_data_mmd.obsm['X_umap'][:,1],
                                       'batch': merged_data_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_mmd_graph_umap,
                legend = "full").set_title("mmd")

########connect the same observations back to each other##########

###pca###
rna_data_mmd_pca = merged_data_mmd_graph_pca[0:len(rna_data)]
atac_data_mmd_pca = merged_data_mmd_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_mmd_pca.iloc[i,0], atac_data_mmd_pca.iloc[i,0]], 
             [rna_data_mmd_pca.iloc[i,1], atac_data_mmd_pca.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_mmd_pca.iloc[iterator,0], 
            rna_data_mmd_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_mmd_pca.iloc[iterator,0], 
            atac_data_mmd_pca.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, mmd")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_mmd_tsne = merged_data_mmd_graph_tsne[0:len(rna_data)]
atac_data_mmd_tsne = merged_data_mmd_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_mmd_tsne.iloc[i,0], atac_data_mmd_tsne.iloc[i,0]], 
             [rna_data_mmd_tsne.iloc[i,1], atac_data_mmd_tsne.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_mmd_tsne.iloc[iterator,0], 
            rna_data_mmd_tsne.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_mmd_tsne.iloc[iterator,0], 
            atac_data_mmd_tsne.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, mmd")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_mmd_umap = merged_data_mmd_graph_umap[0:len(rna_data)]
atac_data_mmd_umap = merged_data_mmd_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_mmd_umap.iloc[i,0], atac_data_mmd_umap.iloc[i,0]], 
             [rna_data_mmd_umap.iloc[i,1], atac_data_mmd_umap.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_mmd_umap.iloc[iterator,0], 
            rna_data_mmd_umap.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_mmd_umap.iloc[iterator,0], 
            atac_data_mmd_umap.iloc[iterator,1],
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, mmd")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

plt.show()

##############################################################################
########evaluate results for data without preprocessing#######################
##############################################################################

neighbors = 5
merged_data_npp_mmd = merged_data_npp.copy()
sc.pp.neighbors(merged_data_npp_mmd, use_rep = 'X_mmd', n_neighbors = neighbors, n_pcs = 50)
sc.tl.leiden(merged_data_npp_mmd)
merged_data_npp_mmd.obsm['X_pca'] = merged_data_npp_mmd.obsm['X_mmd']
sc.tl.umap(merged_data_npp_mmd)
sc.tl.tsne(merged_data_npp_mmd, use_rep = 'X_mmd')
merged_data_npp_mmd.obs['kmeans_mmd'] = skc.KMeans(n_clusters = 2).fit(merged_data_npp_mmd.obsm['X_mmd']).labels_

####control ASW###
merged_data_npp_mmd_asw = skm.silhouette_score(merged_data_npp_mmd.obsm['X_mmd'], merged_data_npp_mmd.obs['batch'])
merged_data_npp_mmd_umap_asw = skm.silhouette_score(merged_data_npp_mmd.obsm['X_umap'], merged_data_npp_mmd.obs['batch'])

###control ARI###
merged_data_npp_mmd_rand = skm.adjusted_rand_score(merged_data_npp_mmd.obs['batch'], merged_data_npp_mmd.obs['kmeans_mmd'])

###control PCA###
merged_data_npp_mmd_graph_pca = pd.DataFrame({'pca-1': merged_data_npp_mmd.obsm['X_pca'][:,0],
                                 'pca-2': merged_data_npp_mmd.obsm['X_pca'][:,1],
                                 'batch': merged_data_npp_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "pca-1", y = "pca-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_harmony_graph_pca,
                legend = "full").set_title("mmd, no pp")

###control tsne###
merged_data_npp_mmd_graph_tsne = pd.DataFrame({'tsne-1': merged_data_npp_mmd.obsm['X_tsne'][:,0],
                                       'tsne-2': merged_data_npp_mmd.obsm['X_tsne'][:,1],
                                       'batch': merged_data_npp_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_mmd_graph_tsne,
                legend = "full").set_title("mmd, no pp")

###control umap###
merged_data_npp_mmd_graph_umap = pd.DataFrame({'umap-1': merged_data_npp_mmd.obsm['X_umap'][:,0],
                                       'umap-2': merged_data_npp_mmd.obsm['X_umap'][:,1],
                                       'batch': merged_data_npp_mmd.obs.batch})
plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_data_npp_mmd_graph_umap,
                legend = "full").set_title("mmd, no pp")
plt.show()

########connect the same observations back to each other##########

###pca###
rna_data_npp_mmd_pca = merged_data_npp_mmd_graph_pca[0:len(rna_data)]
atac_data_npp_mmd_pca = merged_data_npp_mmd_graph_pca[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_mmd_pca.iloc[i,0], atac_data_npp_mmd_pca.iloc[i,0]], 
             [rna_data_npp_mmd_pca.iloc[i,1], atac_data_npp_mmd_pca.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_mmd_pca.iloc[iterator,0], 
            rna_data_npp_mmd_pca.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_mmd_pca.iloc[iterator,0],
            atac_data_npp_mmd_pca.iloc[iterator,1],
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("PCA, mmd no pp")
plt.ylabel("pca-2")
plt.legend(loc = "best")

plt.show()

###tsne###
rna_data_npp_mmd_tsne = merged_data_npp_mmd_graph_tsne[0:len(rna_data)]
atac_data_npp_mmd_tsne = merged_data_npp_mmd_graph_tsne[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_mmd_tsne.iloc[i,0], atac_data_npp_mmd_tsne.iloc[i,0]], 
             [rna_data_npp_mmd_tsne.iloc[i,1], atac_data_npp_mmd_tsne.iloc[i,1]],
             'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_mmd_tsne.iloc[iterator,0], 
            rna_data_npp_mmd_tsne.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_mmd_tsne.iloc[iterator,0], 
            atac_data_npp_mmd_tsne.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("t-SNE, mmd no pp")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
plt.legend(loc = "best")

plt.show()

###umap###
rna_data_npp_mmd_umap = merged_data_npp_mmd_graph_umap[0:len(rna_data)]
atac_data_npp_mmd_umap = merged_data_npp_mmd_graph_umap[len(atac_data):]
iterator = random.sample(range(0,len(rna_data),1), 20)

for i in iterator:
    plt.plot([rna_data_npp_mmd_umap.iloc[i,0], atac_data_npp_mmd_umap.iloc[i,0]], 
             [rna_data_npp_mmd_umap.iloc[i,1], atac_data_npp_mmd_umap.iloc[i,1]],'r-' , linewidth = .2)
    
plt.scatter(rna_data_npp_mmd_umap.iloc[iterator,0], 
            rna_data_npp_mmd_umap.iloc[iterator,1], 
            color = 'b', s = 3, label = "RNA-seq")
plt.scatter(atac_data_npp_mmd_umap.iloc[iterator,0], 
            atac_data_npp_mmd_umap.iloc[iterator,1], 
            color = 'g', s = 3, label = "ATAC-seq")
plt.title("UMAP, mmd no pp")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.legend(loc = "best")

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

###kBET-control###
np.savetxt("dna_atac/results/merged.csv", merged_data.X, delimiter = ",")
np.savetxt("dna_atac/results/merged_npp.csv", merged_data_npp.X, delimiter = ",")


###kBET-harmony###
np.savetxt("dna_atac/results/merged_harmony.csv", merged_data_harmony.obsm['X_harmony'], delimiter = ",")
np.savetxt("dna_atac/results/merged_npp_harmony.csv", merged_data_npp_harmony.obsm['X_harmony'], delimiter = ",")

###kBET-UnionCom###
np.savetxt("dna_atac/results/merged_unioncom.csv", merged_data_unioncom.obsm['X_unioncom'], delimiter = ",")

###kBET-MMD###
np.savetxt("dna_atac/results/merged_mmd.csv", merged_data_mmd.obsm['X_mmd'], delimiter = ",")
np.savetxt("dna_atac/results/merged_npp_mmd.csv", merged_data_npp_mmd.obsm['X_mmd'], delimiter = ",")

###LISI-control###
np.savetxt("dna_atac/results/merged_umap.csv", merged_data.obsm['X_umap'], delimiter = ",")
np.savetxt("dna_atac/results/merged_npp_umap.csv", merged_data_npp.obsm['X_umap'], delimiter = ",")

###LISI-harmony###
np.savetxt("dna_atac/results/merged_harmony_umap.csv", merged_data_harmony.obsm['X_harmony'], delimiter = ",")
np.savetxt("dna_atac/results/merged_npp_harmony_umap.csv", merged_data_npp_harmony.obsm['X_harmony'], delimiter = ",")

###LISI-unioncom###
np.savetxt("dna_atac/results/merged_unioncom_umap.csv", merged_data_unioncom.obsm['X_unioncom'], delimiter = ",")

###LISI-mmd###
np.savetxt("dna_atac/results/merged_mmd_umap.csv", merged_data_mmd.obsm['X_mmd'], delimiter = ",")
np.savetxt("dna_atac/results/merged_npp_mmd_umap.csv", merged_data_npp_mmd.obsm['X_mmd'], delimiter = ",")

###export table of ARI and ASW for each method

merged_data_ARI = np.asarray([merged_data_rand, 
                     merged_data_harmony_rand, 
                     merged_data_unioncom_rand, 
                     merged_data_mmd_rand])

merged_data_ASW = np.asarray([merged_data_asw,
                     merged_data_harmony_asw,
                     merged_data_unioncom_asw,
                     merged_data_mmd_asw])

merged_data_npp_ARI = np.asarray([merged_data_npp_rand,
                              merged_data_npp_harmony_rand,
                              0,
                              merged_data_npp_mmd_rand])

merged_data_npp_ASW = np.asarray([merged_data_npp_asw,
                              merged_data_npp_harmony_asw,
                              0,
                              merged_data_npp_mmd_asw])

merged_ASW_ARI = pd.DataFrame({'ASW_with_preprocessing': merged_data_ASW,
                                 'ARI_with_preprocessing': merged_data_ARI,
                                 'ASW_no_preprocessing': merged_data_npp_ASW,
                                 'ARI_no_preprocessing': merged_data_npp_ARI}, 
                              index = ['control', 'harmony', 'unioncom','mmd'])

merged_ASW_ARI.to_csv("dna_atac/results/ATAC_ASW_ARI.csv")


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

#see metrics.R for rest#