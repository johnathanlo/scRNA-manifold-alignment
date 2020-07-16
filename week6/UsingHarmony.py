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



visualize([data_x_control, data_y_control], control, title = 'control, unintegrated')
visualize([data_x_control, data_y_control], control_h, title = 'control, integrated with harmony')
###no shared features, before
visualize([data_x_unc, data_y_unc], uncorrected_graph, title = 'no correspondence, unintegrated')
###after
visualize([data_x_unc, data_y_unc], corrected_graph, title = 'no correspondence, integrated with harmony')

############evaluation##############
###t-sne###
batch_ids1 = np.tile(0, corrected_graph[0].shape[0])
batch_ids1.shape = (corrected_graph[0].shape[0], 1)
batch_ids2 = np.tile(1, corrected_graph[1].shape[0])
batch_ids2.shape = (corrected_graph[1].shape[0], 1)
batch_ids = np.append(batch_ids1, batch_ids2)
batch_ids.shape = (final[0].shape[0]+final[1].shape[0], 1)
merged_corrected_graphs = np.vstack((corrected_graph[0], corrected_graph[1]))

merged_corrected_graphs_tsne = manifold.TSNE().fit_transform(merged_corrected_graphs)

merged_corrected_graphs_tsne = np.hstack((merged_corrected_graphs_tsne, batch_ids))
merged_corrected_graphs_tsne = pd.DataFrame({'tsne-1': merged_corrected_graphs_tsne[:,0], 
                                 'tsne-2': merged_corrected_graphs_tsne[:,1], 
                                 'batch': merged_corrected_graphs_tsne[:,2]})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "tsne-1", y = "tsne-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_corrected_graphs_tsne,
                legend = "full").set_title("without correspondence, harmony")


###umap###
reducer = umap.UMAP()
merged_corrected_graphs_umap = reducer.fit_transform(merged_corrected_graphs)
merged_corrected_graphs_umap = np.hstack((merged_corrected_graphs_umap, batch_ids))
merged_corrected_graphs_umap = pd.DataFrame({'umap-1': merged_corrected_graphs_umap[:,0], 
                                 'umap-2': merged_corrected_graphs_umap[:,1], 
                                 'batch': merged_corrected_graphs_umap[:,2]})

plt.figure(figsize = (16,10))
sns.scatterplot(x = "umap-1", y = "umap-2", hue = 'batch', 
                palette = sns.color_palette("hls", 2),
                data = merged_corrected_graphs_umap,
                legend = "full").set_title("without correspondence, harmony")

###kBET###
np.savetxt("merged_harmony.csv", merged_corrected_graphs, delimiter = ",")
#see metrics.R#

###LISI###
np.savetxt("merged_harmony_umap.csv", merged_corrected_graphs_umap, delimiter = ",")
#see metrics.R for rest#

###ASW###
merged_corrected_graphs_umap_asw = skm.silhouette_score(merged_corrected_graphs_umap, Donors_merged.obs['batch'])

###ARI###
merged_corrected_graphs_kmeans = skc.KMeans(n_clusters = 2).fit(merged_corrected_graphs_umap).labels_
merged_corrected_graphs_rand = skm.adjusted_rand_score(Donors_merged.obs['batch'], merged_corrected_graphs_kmeans)
