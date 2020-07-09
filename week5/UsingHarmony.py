# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:33:52 2020

@author: jlo
"""

from harmony import harmonize
from unioncom import UnionCom
import scanpy as sc

data_x_control = Donors_merged_control[Donors_merged_control.obs['batch']=='0',:].X
data_y_control = Donors_merged_control[Donors_merged_control.obs['batch']=='1',:].X
control = [data_x_control, data_y_control]
Donors_merged_control_harmony = Donors_merged_control

control_harmonized = harmonize(Donors_merged_control.obsm['X_pca'], Donors_merged_control.obs, batch_key = 'batch')
Donors_merged_control_harmony.obsm['X_harmony'] = control_harmonized
data_x_control_h = Donors_merged_control_harmony.obsm['X_harmony'][Donors_merged_control_harmony.obs['batch']=='0',:]
data_y_control_h = Donors_merged_control_harmony.obsm['X_harmony'][Donors_merged_control_harmony.obs['batch']=='1',:]
control_h = [data_x_control_h, data_y_control_h]

uncorrected = DonorA_sample.concatenate(DonorB_sample, join = 'outer')
data_x_unc = uncorrected[uncorrected.obs['batch']=='0',:].X.toarray()
data_x_unc[np.isnan(data_x_unc)]=0
data_y_unc = uncorrected[uncorrected.obs['batch']=='1',:].X.toarray()
data_y_unc[np.isnan(data_y_unc)]=0
uncorrected_graph = [data_x_unc, data_y_unc]

uncorrected_harmonized = harmonize(uncorrected.obsm['X_pca'], uncorrected.obs, batch_key = 'batch')
uncorrected.obsm['X_harmony'] = uncorrected_harmonized
data_x_corrected = uncorrected.obsm['X_harmony'][uncorrected.obs['batch']=='0',:]
data_y_corrected = uncorrected.obsm['X_harmony'][uncorrected.obs['batch']=='1',:]
corrected_graph = [data_x_corrected, data_y_corrected]



UnionCom.visualize([data_x_control, data_y_control], control)
UnionCom.visualize([data_x_control, data_y_control], control_h)
###no shared features, before
UnionCom.visualize([data_x_unc, data_y_unc], uncorrected_graph)
###after
UnionCom.visualize([data_x_unc, data_y_unc], corrected_graph)