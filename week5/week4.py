# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:44:32 2020

@author: jlo
"""

import itertools
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp 
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt
from harmony import harmonize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.sparse import csr_matrix, find, csgraph
from scipy.spatial import distance_matrix
from scipy.linalg import block_diag
from torchvision import models
from itertools import chain
from torch.autograd import Variable
import torch.nn as nn
import torch
import keras.optimizers
from unioncom import UnionCom

class Project(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Project, self).__init__()
		self.restored = False
		self.input_dim = input_dim
		self.output_dim = output_dim

		num = len(input_dim)
		feature = []

		for i in range(num):
			feature.append(
			nn.Sequential(
			nn.Linear(self.input_dim[i],2*self.input_dim[i]),
			nn.BatchNorm1d(2*self.input_dim[i]),
			nn.LeakyReLU(0.1, True),
			nn.Linear(2*self.input_dim[i],2*self.input_dim[i]),
			nn.BatchNorm1d(2*self.input_dim[i]),
			nn.LeakyReLU(0.1, True),
			nn.Linear(2*self.input_dim[i],self.input_dim[i]),
			nn.BatchNorm1d(self.input_dim[i]),
			nn.LeakyReLU(0.1, True),
			nn.Linear(self.input_dim[i],self.output_dim),
			nn.BatchNorm1d(self.output_dim),
			nn.LeakyReLU(0.1, True),
		))

		self.feature = nn.ModuleList(feature)

		self.feature_show = nn.Sequential(
			nn.Linear(self.output_dim,self.output_dim),
			nn.BatchNorm1d(self.output_dim),
			nn.LeakyReLU(0.1, True),
			nn.Linear(self.output_dim,self.output_dim),
			nn.BatchNorm1d(self.output_dim),
			nn.LeakyReLU(0.1, True),
			nn.Linear(self.output_dim,self.output_dim),
		)

	def forward(self, input_data, domain):
		feature = self.feature[domain](input_data)
		feature = self.feature_show(feature)

		return feature
    
class params():
    epoch_pd = 20000
    epoch_DNN = 200
    epsilon = 0.001
    lr = 0.001
    batch_size = 100
    rho = 10
    log_DNN = 10
    log_pd = 500
    manual_seed = 8888
    delay = 0
    kmax = 20
    beta = 1


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
#################Begin UnionCom######################credit to Kai et al#########
d_x = DonorA.X.shape[1]
n_x = DonorA.X.shape[0]

d_y = DonorB.X.shape[1]
n_y = DonorB.X.shape[0]

data = DonorA.X
obs = list(range(0,len(DonorA.X)))
df = pd.DataFrame(data, columns = DonorA.var_names, index = obs)
K_x = distance_matrix(df.values, df.values)

data = DonorB.X
obs = list(range(0,len(DonorB.X)))
df = pd.DataFrame(data, columns = DonorB.var_names, index = obs)
K_y = distance_matrix(df.values, df.values)

data_x = DonorA_sample.X
data_y = DonorB_sample.X


######get geodesic distances#####
kmin_x = 5 
kmax = 50
nbrs_x = NearestNeighbors(n_neighbors=kmin_x, metric='euclidean', n_jobs=-1).fit(data_x)
knn_x = nbrs_x.kneighbors_graph(data_x, mode='distance')
#check if graph is fully connected
connected = csgraph.connected_components(knn_x, directed=False)[0]

while connected != 1:
	if kmin_x > np.max((kmax, 0.01*len(X))):
		break
	kmin_x += 2
	nbrs_x = NearestNeighbors(n_neighbors=kmin_x, metric='euclidean', n_jobs=-1).fit(data_y)
	knn_x = nbrs_x.kneighbors_graph(data_x, mode='distance')
	connected = csgraph.connected_components(knn_x, directed=False)[0]

#floyd-warshall
dist_x = csgraph.floyd_warshall(knn_x, directed=False)
#replace inf values
dist_max = np.nanmax(dist_x[dist_x != np.inf])
dist_x[dist_x > dist_max] = 2*dist_max


kmin_y = 5 
kmax = 50
nbrs_y = NearestNeighbors(n_neighbors=kmin_y, metric='euclidean', n_jobs=-1).fit(data_y)
knn_y = nbrs_y.kneighbors_graph(data_y, mode='distance')
#check if graph is fully connected
connected = csgraph.connected_components(knn_y, directed=False)[0]

while connected != 1:
	if kmin_y > np.max((kmax, 0.01*len(X))):
		break
	kmin_y += 2
	nbrs_y = NearestNeighbors(n_neighbors=kmin_y, metric='euclidean', n_jobs=-1).fit(data_y)
	knn_y = nbrs_y.kneighbors_graph(data_y, mode='distance')
	connected = csgraph.connected_components(knn_y, directed=False)[0]

#floyd-warshall
dist_y = csgraph.floyd_warshall(knn_y, directed=False)
#replace inf values
dist_max = np.nanmax(dist_y[dist_y != np.inf])
dist_y[dist_y > dist_max] = 2*dist_max

dist = np.array([dist_x, dist_y])
kmin = list([kmin_x, kmin_y])

#########align cells across datasets by matching geometric distance matrices#######

def perplexity(distances, sigmas):
	return calc_perplexity(calc_P(distances, sigmas))

def calc_perplexity(prob_matrix):
	entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
	perplexity = 2 ** entropy
	return perplexity

def calc_P(distances, sigmas=None):
	if sigmas is not None:
		two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
		return softmax(distances / two_sig_sq)
	else:
		return softmax(distances)
    
def softmax(D, diag_zero=True):
	# e_x = np.exp(D)
	e_x = np.exp(D - np.max(D, axis=1).reshape([-1, 1]))
	if diag_zero:
		np.fill_diagonal(e_x, 0)
	e_x = e_x + 1e-15
	return e_x / e_x.sum(axis=1).reshape([-1,1])  
    
def binary_search(eval_fn, target ,tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
	for i in range(max_iter):
		guess = (lower + upper) /2.
		val = eval_fn(guess)
		if val > target:
			upper = guess
		else:
			lower = guess
		if np.abs(val - target) <= tol:
			break
	return guess

def p_conditional_to_joint(P):
	return (P + P.T) / (2. * P.shape[0])

def init_model(net, device, restore):
	if restore is not None and os.path.exits(restore):
		net.load_state_dict(torch.load(restore))
		net.restored = True
		print("Restore model from: {}".format(os.path.abspath(restore)))
	else:
		print("No trained model, train UnionCom from scratch.")

	if torch.cuda.is_available():
		cudnn.benchmark =True
		net.to(device)

	return net

p = []
##negative square distances
neg_dist_x = -dist[0]
neg_dist_y = -dist[1]
###find sigmas
sigmas_x = []
for i in range(neg_dist_x.shape[0]):
	eval_fn = lambda sigma: perplexity(neg_dist_x[i:i+1, :], np.array(sigma))
	correct_sigma = binary_search(eval_fn, kmin[0])
	sigmas_x.append(correct_sigma)
sigmas_x = np.array(sigmas_x)

sigmas_y = []
for i in range(neg_dist_y.shape[0]):
	eval_fn = lambda sigma: perplexity(neg_dist_y[i:i+1, :], np.array(sigma))
	correct_sigma = binary_search(eval_fn, kmin[1])
	sigmas_y.append(correct_sigma)    
sigmas_y = np.array(sigmas_y)

p_cond_x = calc_P(neg_dist_x, sigmas_x)
p_cond_y = calc_P(neg_dist_y, sigmas_y)

p_x = p_conditional_to_joint(p_cond_x)
p_y = p_conditional_to_joint(p_cond_y)

p = [p_x, p_y]

#############Project unmatched features into common space#################
input_dims = [data_x.shape[1], data_y.shape[1]]
output_dim = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Project(input_dims, output_dim)
project_net = init_model(net, device, restore=None)


result = UnionCom.train(project_net, params, [data_x,data_y], dist, p, device)

final = UnionCom.fit_transform([data_x,data_y], datatype=None, epoch_pd=1000, epoch_DNN=100, epsilon=0.001, 
lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=666, delay=0, 
beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=32, test=False)

UnionCom.visualize([data_x, data_y], final)

################################MMD-RESNET#######################credit Uri Shaham

import os.path
import keras.optimizers
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
from keras.layers import Input, Dense, merge, Activation, add
from keras.models import Model
from keras import callbacks as cb
import numpy as np
import matplotlib
from keras.layers.normalization import BatchNormalization
#detect display
import os
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

import CostFunctions as cf
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
import ScatterHist as sh
from keras import initializers
from numpy import genfromtxt
import sklearn.preprocessing as prep
import tensorflow as tf
import keras.backend as K


# configuration hyper parameters
denoise = False # whether or not to train a denoising autoencoder to remove the zeros
keepProb=.8

# AE confiduration
ae_encodingDim = 25
l2_penalty_ae = 1e-2 

#MMD net configuration
mmdNetLayerSizes = [25, 25]
l2_penalty = 1e-2
#init = lambda shape, name:initializations.normal(shape, scale=.1e-4, name=name)
#def my_init (shape):
#    return initializers.normal(stddev=.1e-4)
#my_init = 'glorot_normal'

#######################
###### read data ######
#######################
# we load two CyTOF samples 

 
source = data_x
target = data_y

# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
source = dh.preProcessCytofData(source) 

numZerosOK=1
toKeepS = np.sum((source==0), axis = 1) <=numZerosOK
print(np.sum(toKeepS))
toKeepT = np.sum((target==0), axis = 1) <=numZerosOK
print(np.sum(toKeepT))

inputDim = target.shape[1]

if denoise:
    trainTarget_ae = np.concatenate([source[toKeepS], target[toKeepT]], axis=0)
    np.random.shuffle(trainTarget_ae)
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb, size = trainTarget_ae.shape)
    input_cell = Input(shape=(inputDim,))
    encoded = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(input_cell)
    encoded1 = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(encoded)
    decoded = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty_ae))(encoded1)
    autoencoder = Model(input=input_cell, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.fit(trainData_ae, trainTarget_ae, epochs=500, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=25,  mode='auto')])    
    source = autoencoder.predict(source)
    target = autoencoder.predict(target)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source) 
target = preprocessor.transform(target)    

#############################
######## train MMD net ######
#############################


calibInput = Input(shape=(inputDim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2) 
block1_output = add([block1_w2, calibInput])
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2) 
block2_output = add([block2_w2, block1_output])
block3_bn1 = BatchNormalization()(block2_output)
block3_a1 = Activation('relu')(block3_bn1)
block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1) 
block3_bn2 = BatchNormalization()(block3_w1)
block3_a2 = Activation('relu')(block3_bn2)
block3_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2) 
block3_output = add([block3_w2, block2_output])

calibMMDNet = Model(inputs=calibInput, outputs=block3_output)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 150.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

#train MMD net
optimizer = keras.optimizers.rmsprop(lr=0.0)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block3_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
K.get_session().run(tf.global_variables_initializer())

sourceLabels = np.zeros(source.shape[0])
calibMMDNet.fit(source,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate, mn.monitorMMD(source, target, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto')])

##############################
###### evaluate results ######
##############################

calibratedSource = calibMMDNet.predict(source)

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after = pca.transform(calibratedSource)

# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)
 
# save models
autoencoder.save(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_DAE.h5'))                 
calibMMDNet.save_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 6 00:17:31 2020
@author: urixs
"""

import numpy as np
from sklearn import decomposition
import argparse
from itertools import count
import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler



from matplotlib.ticker import NullFormatter
from torch.autograd import Variable

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
                    default=5,
                    help="Number of epochs without improvement before stopping")
parser.add_argument("--save_dir",
                    type=str,
                    default='./calibrated_data',
                    help="Directory for calibrated data")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# =                                   Dataset                                  =
# ==============================================================================

sample1 = DonorA_sample.X[:,0:1300]
sample2 = DonorB_sample.X[:,0:1300]

sample1_tensor = torch.Tensor(sample1)
sample1_dataset = torch.utils.data.TensorDataset(sample1_tensor)

sample2_tensor = torch.Tensor(sample2)
sample2_dataset = torch.utils.data.TensorDataset(sample2_tensor)

sample1_loader = torch.utils.data.DataLoader(sample1_dataset,
                                          batch_size=128,
                                          shuffle=True)

sample2_loader = torch.utils.data.DataLoader(sample2_dataset,
                                          batch_size=128,
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
     
def main():
    
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
    
    
if __name__ == '__main__':
    main()