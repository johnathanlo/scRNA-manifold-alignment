
 _______  _______  _______  ______   _______  _______ 
(  ____ )(  ____ \(  ___  )(  __  \ (       )(  ____ \
| (    )|| (    \/| (   ) || (  \  )| () () || (    \/
| (____)|| (__    | (___) || |   ) || || || || (__    
|     __)|  __)   |  ___  || |   | || |(_)| ||  __)   
| (\ (   | (      | (   ) || |   ) || |   | || (      
| ) \ \__| (____/\| )   ( || (__/  )| )   ( || (____/\
|/   \__/(_______/|/     \|(______/ |/     \|(_______/

Johnathan Lo, 2020

$This set of data, code, and results, constitute a brief introduction to perfo-$
$rming preprocessing, batch correction, and batch correction evaluation for si-$
$ngle cell sequencing data. This readme will serve as a guide to navigating ar-$
$ound the files and folders located in this directory, and a repository for re-$
$ferences. For more specific instructions on running and understanding the pro-$
$cedures, consult the code comments and accompanying PowerPoint materials. Code$ 
$requires Python 3.7.7 and R 3.6.1, in addition to other various libraries/mod-$
$les listed below.										                       $
$                                                                              $
$==============================================================================$
$==============================================================================$
$                                                                              $
$Directory:                                                                    $
$	preproc_intro 	#This folder contains tutorials on running standard prepro-$
$					cessing procedures in Python.                              $
$		data        #Contains the data files for running the tutorials.        $
$       src         #Code for tutorials.                                       $
$       intro.pptx	#PPT guide to accompany code.                              $
$	                                                                           $
$	donors_merged	#This folder contains tutorials for running preprocessing  $
$					#and data integration (batch correction) methods for data  $
$					#in the same feature space but where the correspondence be-$
$					#tween features is not known.							   $
$		data		#Same as above.											   $
$		src			#Same as above.											   $
$		donors.pptx	#PPT guide.												   $
$																			   $
$	dna_atac		#This folder contains tutorials for running preprocessing  $
$					#and data integration for data where the feature space is  $
$					#different but observations are identical. 				   $
$		data		#Same as above.											   $
$		src			#Same as above.											   $
$		atac.pptx	#PPT guide.												   $																			   
$																			   $
$==============================================================================$
$==============================================================================$
$                                                                              $
$Methods:																	   $
$		Preprocessing - Scanpy 1.5.1										   $
$			Installation - pip install scanpy								   $
$			Reference - Wolf, F., Angerer, P. & Theis, F. SCANPY: large-scale  $ 
$						single-cell gene expression data analysis. Genome Biol $
$						19, 15 (2018).                                         $
$						https://doi.org/10.1186/s13059-017-1382-0              $
$																			   $
$		Data integration method 1 - Manifold Alignment without Correspondence  $
$			Installation - NA - refer to code								   $
$			Reference - C. Wang and S. Mahadevan. Manifold alignment without   $
$						correspondence. In IJCAI, pages 1273–1278, Pasadena,   $
$						CA, USA, July 2009.                                    $
$																			   $
$		Data integration method 2 - Harmony	v.0.1.4							   $
$			Installation - pip install harmony-pytorch						   $
$			Reference - Korsunsky, I., Millard, N., Fan, J. et al. Fast, 	   $
$						sensitive and accurate integration of single-cell data $
$						with Harmony. Nat Methods 16, 1289–1296 (2019).        $
$						https://doi.org/10.1038/s41592-019-0619-0			   $
$																			   $
$		Data integration method 3 - UnionCom v.0.2.1						   $
$			Installation - pip install unioncom								   $
$			Reference - Kai Cao, Xiangqi Bai, Yiguang Hong, Lin Wan,		   $ 
$						Unsupervised topological alignment for single-cell     $
$						multi-omics integration, Bioinformatics, Volume 36,    $
$						Issue Supplement_1, July 2020, Pages i48–i56,          $
$						https://doi.org/10.1093/bioinformatics/btaa443         $
$																			   $
$		Data integration method 4 - MMD-ResNet								   $
$			Installation - Not on PyPi, either download from                   $
$							https://github.com/ushaham/batchEffectRemoval2020  $
$							or refer to implementation in provided code. 	   $
$			Reference - Uri Shaham, Kelly P Stanton, Jun Zhao, Huamin Li,	   $ 
$						Khadir Raddassi, Ruth Montgomery, Yuval Kluger,        $
$						Removal of batch effects using distribution-matching   $
$						residual networks, Bioinformatics, Volume 33, Issue 16,$
$						15 August 2017, Pages 2539–2546, 					   $
$						https://doi.org/10.1093/bioinformatics/btx196          $
$																			   $
$		Evaluation metric 1 - ASW (average silhouette width)				   $
$			Installation - Included in scikit-learn.metrics as silhouette_score$
$			Reference - Rousseeuw PJ. Silhouettes: a graphical aid to the	   $ 
$						interpretation and validation of cluster analysis.     $
$						J Comput Appl Math. 1987;20:53–65 Available from:      $
$						http://www.sciencedirect.com/science/article/pii/      $
$						0377042787901257.									   $
$																			   $
$		Evaluation metric 2 - ARI (adjusted Rand index)						   $
$			Installation - Included in scikit-learn.metrics as				   $
$							adjusted_rand_score.							   $
$			Reference - Hubert L, Arabie P. Comparing partitions. J Classif.   $
$						1985;2:193–218.										   $
$																			   $
$		Evaluation metric 3 - PCA (principal component analysis)			   $
$			Installation - included in scikit-learn as well as Scanpy. Consult $
$							documentation for implementation.				   $
$			Reference - Pearson, K. (1901). "On Lines and Planes of Closest Fit$
$		 				to Systems of Points in Space". Philosophical Magazine.$
$						2 (11): 559–572. doi:10.1080/14786440109462720.		   $
$																			   $
$		Evaluation metric 4 - t-SNE (t-distributed stochastic neighbor 		   $
$									embedding)								   $
$			Installation - included in ScanPy. For additional visualization    $
$						   options, consult code.							   $
$			Reference - van der Maaten L, Hinton G. Visualizing data using 	   $
$						t-SNE; 2008.										   $
$																			   $
$		Evaluation metric 5 - UMAP (uniform manifold approximation)			   $
$			Installation - included in ScanPy. For additional visualization    $
$						   options, consult code.							   $
$			Reference - McInnes L, Healy J, Melville J. UMAP: Uniform Manifold $
$						Approximation and Projection for dimension reduction.  $
$						arXiv. 2018;1802:arXiv Prepr arXiv180203426.		   $
$																			   $
$		Evaluation metric 6 - kBET (knn batch effect test)					   $
$			Installation - Implemented in R as kBET. install.packages("kBET")  $
$			Reference - Buttner M, Miao Z, Wolf FA, Teichmann SA, Theis FJ. A  $
$						test metric for assessing single-cell RNA-seq batch    $
$						correction. Nat Methods. 2019;16:43–9.				   $
$																			   $
$		Evaluation metric 7 - LISI (local inverse Simpson's index)			   $
$			Installation - Implemented in R as lisi. install.packages("lisi")  $
$			Reference - Korsunsky I, Millard N, Fan J, Slowikowski K, Zhang F, $
$						Wei K, Baglaenko Y, Brenner M, Loh P-r, Raychaudhuri S.$
$						Fast, sensitive and accurate integration of single-cell$ 
$						data with Harmony. Nature Methods; 2019.               $
$						https://doi.org/10.1038/s41592-019-0619-0. Accessed 1  $
$						Mar 2019.											   $
$																			   $
$==============================================================================$
$==============================================================================$
$																			   $
$Additional resources:														   $
$	1. Tran, H.T.N., Ang, K.S., Chevrier, M. et al. A benchmark of batch-effect$ 
$		correction methods for single-cell RNA sequencing data. Genome Biol 21,$
$		12 (2020). https://doi.org/10.1186/s13059-019-1850-9				   $
$																			   $
$==============================================================================$
$==============================================================================$