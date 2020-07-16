setwd("~/Summer20/HUANG/data")
library(lisi)
library(kBET)
library(dplyr)
library(tidyr)
library(ggplot2)
DonorA <- read.table("DonorA.csv", sep = ",")
DonorB <- read.table("DonorB.csv", sep = ",")
DonorA_sample <- read.table("DonorA_sample.csv", sep = ",")
DonorB_sample <- read.table("DonorB_sample.csv", sep = ",")
Donors_merged <- read.table("Donors_merged.csv", sep = ",")
Donors_merged_control <- read.table("Donors_merged_control.csv", sep = ",")
DonorA_umap <- read.table("DonorA_umap.csv", sep = ",")
DonorB_umap <- read.table("DonorB_umap.csv", sep = ",")
DonorA_sample_umap <- read.table("DonorA_sample_umap.csv", sep = ",")
DonorB_sample_umap <- read.table("DonorB_sample_umap.csv", sep = ",")
Donors_merged_umap <- read.table("Donors_merged_umap.csv", sep = ",")
Donors_merged_control_umap <- read.table("Donors_merged_control_umap.csv", sep = ",")
Donors_merged_unioncom <- read.table("merged_unioncom.csv", sep = ",")
Donors_merged_unioncom_umap <- read.table("merged_unioncom_umap.csv", sep = ",")
Donors_merged_harmony <- read.table("merged_harmony.csv", sep = ",")
Donors_merged_harmony_umap <- read.table("merged_harmony_umap.csv", sep = ",")

############kBET
##controls
batch_ids <- c(rep(0, 290), rep(1, 775))
kBET(Donors_merged, batch_ids)
kBET(Donors_merged_control, batch_ids)
kBET(DonorA, c(rep(0,290), rep(1,2553)))

##integrated with UnionCom##

kBET(Donors_merged_unioncom, batch_ids)

##integrated with harmony##

kBET(Donors_merged_harmony, batch_ids)

#################LISI
batch_ids_lisi <- data.frame(batch = batch_ids)

lisi_res <- compute_lisi(Donors_merged_umap, batch_ids_lisi, c('batch'))
Donors_merged_umap %>% 
  cbind(lisi_res) %>% 
  dplyr::sample_frac(1L, FALSE) %>% 
  tidyr::gather(key, lisi_value, batch) %>% 
  ggplot(aes(V1, V2, color = lisi_value)) + geom_point(shape = 21) + 
  facet_wrap(~key)

lisi_res <- compute_lisi(Donors_merged_control_umap, batch_ids_lisi, c('batch'))
Donors_merged_control_umap %>% 
  cbind(lisi_res) %>% 
  dplyr::sample_frac(1L, FALSE) %>% 
  tidyr::gather(key, lisi_value, batch) %>% 
  ggplot(aes(V1, V2, color = lisi_value)) + geom_point(shape = 21) + 
  facet_wrap(~key)

##integrated with unioncom##
lisi_res <- compute_lisi(Donors_merged_unioncom_umap, batch_ids_lisi, c('batch'))
Donors_merged_unioncom_umap %>% 
  cbind(lisi_res) %>% 
  dplyr::sample_frac(1L, FALSE) %>% 
  tidyr::gather(key, lisi_value, batch) %>% 
  ggplot(aes(V1, V2, color = lisi_value)) + geom_point(shape = 21) + 
  facet_wrap(~key)

##integrated with harmony##
lisi_res <- compute_lisi(Donors_merged_harmony_umap, batch_ids_lisi, c('batch'))
Donors_merged_harmony_umap %>% 
  cbind(lisi_res) %>% 
  dplyr::sample_frac(1L, FALSE) %>% 
  tidyr::gather(key, lisi_value, batch) %>% 
  ggplot(aes(V1, V2, color = lisi_value)) + geom_point(shape = 21) + 
  facet_wrap(~key)





