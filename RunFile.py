##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Fri Jul 28 10:44:38 2017
#
#@author: aki.nikolaidis
#"""
#
#import BASC
#from BASC import *
import utils
import os
import numpy as np
import scipy.stats
from os.path import expanduser
from basc_workflow_runner import run_basc_workflow

home = expanduser("~")
proc_mem= [3,6]




subject_file_list= [home + '/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/PyBASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/PyBASC/sample_data/sub3/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/PyBASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/PyBASC/sample_data/sub2/Func_Quarter_Res.nii.gz']

#'/Users/aki.nikolaidis/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz'

roi_mask_file= home + '/git_repo/PyBASC/masks/LC_Quarter_Res.nii.gz'
dataset_bootstraps=5
timeseries_bootstraps=5
n_clusters_list=[2,3]
output_sizes=[5,10]
bootstrap_list=list(range(0,dataset_bootstraps))
cross_cluster=True
roi2_mask_file= home + '/git_repo/PyBASC/masks/RC_Quarter_Res.nii.gz'
affinity_threshold= [0.8] * len(subject_file_list)
ism_gsm_stability=[]
ind_clust_stab_mat=[]
ind_clust_stab_summary=[[1, 2, 3, 4, 5]]
run=True

for n_clusters in n_clusters_list:
    for output_size in output_sizes:
        out_dir= home + '/PyBASC_outputs/multi_set_affinity_FixedTest14/dim_' + str(output_size) + '_' + str(n_clusters) + '_clusters'
        PyBASC_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)
        ism_gsm_stability.append(np.load(out_dir + '/workflow_output/ism_gsm_corr_file/ism_gsm_corr.npy'))
        ind_clust_stab_mat = np.load(out_dir + '/workflow_output/ind_group_cluster_stability_set/ind_group_cluster_stability_set.npy')
        ind_clust_stab_summary=np.concatenate((ind_clust_stab_summary, np.array([[n_clusters, output_size, ind_clust_stab_mat.mean(), scipy.stats.variation(ind_clust_stab_mat).mean(), (ind_clust_stab_mat.mean() - scipy.stats.variation(ind_clust_stab_mat).mean())]])))
        
    print('saving files: ism_gsm_stability')
    ism_gsm_stability_file=os.path.join(out_dir, 'ism_gsm_stability_'+ str(n_clusters)+ '.npy')
    np.save(ism_gsm_stability_file, ism_gsm_stability)
    ism_gsm_stability=[]

print('saving files: ind_clust_stab_summary')
ind_clust_stab_summary_file=os.path.join(out_dir, 'ind_clust_stab_summary.npy')
np.save(ind_clust_stab_summary_file, ind_clust_stab_summary)
