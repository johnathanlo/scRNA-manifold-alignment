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
sc.pp.neighbors(Donors_merged, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset
sc.pp.neighbors(Donors_merged_control, n_neighbors=neighbors, n_pcs=40)#increase # neighbors to account for larger dataset


sc.tl.umap(DonorA)
sc.tl.umap(DonorB)
sc.tl.umap(Donors_merged)
sc.tl.umap(Donors_merged_control)
sc.pl.umap(DonorA, color = ['CST3'])
sc.pl.umap(DonorB, color = ['CST3'])
sc.pl.umap(Donors_merged_control, color = ['CST3'])

sc.pl.pca(DonorA)
sc.pl.pca(DonorB)
sc.pl.pca(DonorA_sample)
sc.pl.pca(DonorB_sample)
sc.pl.pca(Donors_merged, color = 'batch')
sc.pl.pca(Donors_merged_control, color = 'batch')