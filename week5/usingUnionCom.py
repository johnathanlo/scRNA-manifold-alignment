# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:30:34 2020

@author: jlo
"""
import numpy as np
from unioncom import UnionCom
from scipy.sparse import csr_matrix, find, linalg

d_x = DonorA_sample.X.shape[1]
n_x = DonorA_sample.X.shape[0]

d_y = DonorB_sample.X.shape[1]
n_y = DonorB_sample.X.shape[0]


data_x_control = Donors_merged_control[Donors_merged_control.obs['batch']=='0',:].X
data_y_control = Donors_merged_control[Donors_merged_control.obs['batch']=='1',:].X
control = [data_x_control, data_y_control]

uncorrected = DonorA_sample.concatenate(DonorB_sample, join = 'outer')
data_x_unc = uncorrected[uncorrected.obs['batch']=='0',:].X.toarray()
data_x_unc[np.isnan(data_x_unc)]=0
data_y_unc = uncorrected[uncorrected.obs['batch']=='1',:].X.toarray()
data_y_unc[np.isnan(data_y_unc)]=0
uncorrected = [data_x_unc, data_y_unc]

data = DonorA_sample.X
obs = list(range(0,len(DonorA_sample.X)))
df = pd.DataFrame(data, columns = DonorA_sample.var_names, index = obs)
K_x = distance_matrix(df.values, df.values)

data = DonorB_sample.X
obs = list(range(0,len(DonorB_sample.X)))
df = pd.DataFrame(data, columns = DonorB_sample.var_names, index = obs)
K_y = distance_matrix(df.values, df.values)

data_x = DonorA_sample.X
data_y = DonorB_sample.X


final = UnionCom.fit_transform([data_x,data_y], datatype=None, epoch_pd=1000, epoch_DNN=100, epsilon=0.001, 
lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=666, delay=0, 
beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=32, test=False)

###control
UnionCom.visualize([data_x_control, data_y_control], control)
###before
UnionCom.visualize([data_x_unc, data_y_unc], uncorrected)
###after
UnionCom.visualize([data_x, data_y], final)