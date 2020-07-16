# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:30:34 2020

@author: jlo
"""
import numpy as np
from unioncom import UnionCom
from scipy.sparse import csr_matrix, find, linalg
from sklearn import manifold

import seaborn as sns
import pandas as pd
import umap

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
beta=1, kmax=20, distance = 'geodesic', project='tsne', output_dim=50, test=False)

###control
visualize([data_x_control, data_y_control], control, title = 'control')
###before
visualize([data_x_unc, data_y_unc], uncorrected, title = 'no correspondence, uncorrected')
###after
visualize([data_x, data_y], final, title = 'no correspondence, integrated with UnionCom')


############evaluation#############
###t-sne###
batch_ids1 = np.tile(0, final[0].shape[0])
batch_ids1.shape = (final[0].shape[0], 1)
batch_ids2 = np.tile(1, final[1].shape[0])
batch_ids2.shape = (final[1].shape[0], 1)
batch_ids = np.append(batch_ids1, batch_ids2)
batch_ids.shape = (final[0].shape[0]+final[1].shape[0], 1)
merged_final = np.concatenate(final[0], final[1])

merged_final_tsne = manifold.TSNE().fit_transform(merged_final)

merged_final_tsne = np.hstack((merged_final_tsne, batch_ids))
merged_final_tsne = pd.DataFrame({'tsne-1': merged_final_tsne[:,0], 
                                 'tsne-2': merged_final_tsne[:,1], 
                                 'batch': merged_final_tsne[:,2]})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_final_tsne,
                legend = "full").set_title("without correspondence, unioncom")


###umap###
reducer = umap.UMAP()
merged_final_umap = reducer.fit_transform(merged_final)
merged_final_umap = np.hstack((merged_final_umap, batch_ids))
merged_final_umap = pd.DataFrame({'umap-1': merged_final_umap[:,0], 
                                 'umap-2': merged_final_umap[:,1], 
                                 'batch': merged_final_umap[:,2]})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_final_umap,
                legend = "full").set_title("without correspondence, unioncom")

###kBET###
np.savetxt("merged_unioncom.csv", merged_final, delimiter = ",")
#see metrics.R#

###LISI###
np.savetxt("merged_unioncom_umap.csv", merged_final_umap, delimiter = ",")
#see metrics.R for rest#

###ASW###
merged_unioncom_umap_asw = skm.silhouette_score(merged_final_umap, Donors_merged.obs['batch'])

###ARI###
Donors_merged_unioncom_kmeans = skc.KMeans(n_clusters = 2).fit(merged_final_umap).labels_
Donors_merged_unioncom_rand = skm.adjusted_rand_score(Donors_merged.obs['batch'], Donors_merged_unioncom_kmeans)
