# -*- coding: utf-8 -*-

###dataset can be downloaded at http://cf.10xgenomics.com/samples/cell-exp/3.1.0/5k_pbmc_NGSC3_aggr/5k_pbmc_NGSC3_aggr_filtered_feature_bc_matrix.tar.gz
###full tutorial can be found at https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\jlo\Documents\Summer20\HUANG\Data\filtered_feature_bc_matrix")
sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor = 'white')
results_file = "write/pbmc5k.h5ad" ##make sure that 'write' directory exists

adata = sc.read_10x_mtx(path = '',var_names = 'gene_symbols', cache = True) ##this step will take some time - make sure path is in the folder containing mex data
adata.var_names_make_unique()

##preliminary plot of genes by fraction of total counts
sc.pl.highest_expr_genes(adata, n_top=20,)###note large fraction of MT genes
##some interesting summary statistics
adata_avgcountpercell = adata.X.sum()/len(adata.obs)#avg count per cell
adata_avgcountpergene = adata.X.sum()/len(adata.var)
adata_genetots = adata.X.sum(axis = 0)
adata_celltots = adata.X.sum(axis = 1)
plt.hist(adata_genetots, bins = 'auto')
plt.hist(adata_celltots, bins = 'auto')

sc.pp.filter_cells(adata, min_genes = 200)
sc.pp.filter_genes(adata, min_cells=5) ##increased minimum cells for a gene compared to tutorial to account for larger number of cells

adata.var['mt'] = adata.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error
sc.pp.calculate_qc_metrics(adata, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)

##violin plots of number of genes, total counts, and pct counts MT for each cell/barcode
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .2, multi_panel = True)

##perform QC on high MT counts
sc.pl.scatter(adata, x = 'total_counts', y = 'pct_counts_mt')#visualize - note that the mt percentage is on average higher for this dataset, and there appears to be a clear elbow starting at around 18
adata = adata[adata.obs.pct_counts_mt<16, :] #filter
sc.pl.scatter(adata, x = 'total_counts', y = 'pct_counts_mt')#visualize after

##perform QC on high gene counts
sc.pl.scatter(adata, x = 'total_counts', y = 'n_genes_by_counts')#visualize
adata = adata[adata.obs.n_genes_by_counts <6000, :]#filter
sc.pl.scatter(adata, x = 'total_counts', y = 'n_genes_by_counts')#visualize after

##total-count normalization data
sc.pp.normalize_total(adata, target_sum = 2e4) #note that in previous graph, we see that most cells have count <20,000

##log-transform data
sc.pp.log1p(adata)

##extract HVGs
sc.pp.highly_variable_genes(adata, min_mean = .0125, max_mean = 3, min_disp = .5)


##plot HVGs
sc.pl.highly_variable_genes(adata)

##freeze adata
adata.raw = adata

##filter for HVGs
adata = adata[:, adata.var.highly_variable]

#regress out count and mt percentage effects
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

##scale to unit variance
sc.pp.scale(adata, max_value = 10)

#####Pre-processing finished#####

#####Begin PCA#####
sc.tl.pca(adata, svd_solver='arpack')

#visualize 2 biggest components
sc.pl.pca(adata, color = 'CST3')

#Save results
adata.write(results_file)

#####Neighborhood graph and clustering#####

#compute graph
sc.pp.neighbors(adata, n_neighbors=50, n_pcs=40)#increase # neighbors to account for larger dataset

#embed graph
sc.tl.umap(adata)

#plot graph
sc.pl.umap(adata, color = ['CST3', 'NKG7', 'PPBP'])
sc.pl.umap(adata, color = ['MALAT1'])

#cluster graph using Leiden clustering
sc.tl.leiden(adata)#must have leidenalg installed

#plot clusters
sc.pl.umap(adata, color = ['leiden', 'NKG7', 'MALAT1'])

#####find marker genes#####

#rank genes by leiden
sc.tl.rank_genes_groups(adata, 'leiden', method = 't-test')

#graph rankings
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)#note that only for a few clusters there is a distinct marker gene

#save results
adata.write(results_file)

#rank by wilcoxon
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

#rank by logistc regression
sc.tl.rank_genes_groups(adata, 'leiden', method='logreg', )
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

#show top 10 ranked genes per cluster
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)

##comparing clusters 0 and 1
sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)


#######Harmony#######pip install harmony-pytorch
from harmony import harmonize
import os
os.chdir(r"C:\Users\jlo\Documents\Summer20\HUANG\Data")
###load datasets###
adata_293t = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\filtered_matrices_mex_293t\hg19',var_names = 'gene_symbols', cache = True)
adata_jurkat = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\filtered_matrices_mex_jurkat\hg19',var_names = 'gene_symbols', cache = True)
adata_5050 = sc.read_10x_mtx(path = r'C:\Users\jlo\Documents\Summer20\HUANG\Data\filtered_matrices_mex_jurkat_293t\hg19',var_names = 'gene_symbols', cache = True)
adata_merged = adata_293t.concatenate(adata_jurkat, adata_5050)
results_file_merged = "293t-jurkat-5050-merge/write/merge.h5ad" ##make sure that 'write' directory exists
adata_merged.var_names_make_unique()##not sure if necessary

###conduct some basic preproc###
sc.pl.highest_expr_genes(adata_merged, n_top=20,)
sc.pp.filter_cells(adata_merged, min_genes = 200)
sc.pp.filter_genes(adata_merged, min_cells=5)
adata_merged.var['mt'] = adata_merged.var_names.str.startswith('MT-')###not sure why spyder marks this as a syntax error
sc.pp.calculate_qc_metrics(adata_merged, qc_vars = ['mt'], percent_top = None, log1p=False, inplace = True)
sc.pl.violin(adata_merged, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter = .4, multi_panel = True)
#QC for burst cells
sc.pl.scatter(adata_merged, x = 'total_counts', y = 'pct_counts_mt')#visualize - note that the mt percentage is on average higher for this dataset, and there appears to be a clear elbow starting at around 18
adata_merged = adata_merged[adata_merged.obs.pct_counts_mt<8, :] #filter
sc.pl.scatter(adata_merged, x = 'total_counts', y = 'pct_counts_mt')#visualize after
#QC for doublets
sc.pl.scatter(adata_merged, x = 'total_counts', y = 'n_genes_by_counts')#visualize
adata_merged = adata_merged[adata_merged.obs.n_genes_by_counts <5200, :]
sc.pl.scatter(adata_merged, x = 'total_counts', y = 'n_genes_by_counts')
#normalize
sc.pp.normalize_total(adata_merged, target_sum = 2.5e4)
#logtransform
sc.pp.log1p(adata_merged)
#HVG filtering
sc.pp.highly_variable_genes(adata_merged, min_mean = .0125, max_mean = 3, min_disp = .5)
sc.pl.highly_variable_genes(adata_merged)
adata_merged = adata_merged[:, adata_merged.var.highly_variable]
sc.pp.regress_out(adata_merged, ['total_counts', 'pct_counts_mt'])
##scale to unit variance
sc.pp.scale(adata_merged, max_value = 10)
###PCA
sc.tl.pca(adata_merged, svd_solver='arpack')
sc.pl.pca(adata, color = 'RPS2')
#save
adata_merged.write(results_file_merged)

#do some plotting
sc.pl.pca(adata_merged, color = 'batch')

###run harmony
adata_harmony = harmonize(adata_merged.obsm['X_pca'], adata_merged.obs, batch_key = 'batch')
adata_merged.obsm['X_harmony'] = adata_harmony

##plot corrected data
sc.pl.pca(adata_merged, color )