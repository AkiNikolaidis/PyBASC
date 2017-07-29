BASC: Bootstrap Analysis of Stable Clusters
============================================================

A multi-level, open-source, Nipype-based, parcellation pipeline for preprocessed functional MRI data. Designed for use by both novice and expert users, BASC allows individual to find reproducible and reliable clusters at the individual and group level.

Website
-------


Installation
------------


Documentation
-------------
Inputs:

1- Subject file
2- roi_mask_file:
3- dataset_boostraps:
4- n_clusters
5- output_size
6- bootstrap_list= 
7- cross cluster
8- roi2_mask_file
9- cbb_block_size
10-affinity_threshold
11-out_dir
12-run

Outputs:

1- individual_stabilty_matrix- a voxelwise stability matrix of the clustering solution for each individual
2- group_stability_matrix- a voxelwise stability matrix for all subjects
3- gsmap - an voxel by cluster matrix with average within cluster

Upcoming updates:
