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
import PyBASC
from PyBASC.__main__ import main, run_PyBASC
import __init__
import utils
import os
import gc
import numpy as np
import nibabel as nb

import scipy.stats
from os.path import expanduser
from basc_workflow_runner import run_basc_workflow
from basc import save_igcm_nifti, create_group_cluster_maps, ism_nifti, gsm_nifti
home = expanduser("~")
proc_mem= [2,4]


subject_file_list = ['/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_0corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_1corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_2corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_3corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_4corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_5corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_6corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_7corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_8corr_0.05_noise_2_TRs_100.nii.gz',
                     '/Users/aki.nikolaidis/git_repo/PyBASC/SimData4/sub_9corr_0.05_noise_2_TRs_100.nii.gz']


#roi_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Full_BG_Sim_3mm.nii.gz'
#roi2_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Yeo7_3mmMasks/Yeo_2_3mm.nii.gz'

roi_mask_file='masks/Full_BG_Sim_3mm.nii.gz'


roi2_mask_file='masks/Yeo7_3mmMasks/Yeo_2_3mm.nii.gz' #


dataset_bootstrap_list=[2]#,10,30,100]
timeseries_bootstrap_list=[5]#,10,30,100]
similarity_metric_list=['correlation'] #['correlation','euclidean','cityblock', 'cosine']
blocklength_list=[1]#[0.5,1,2]
n_clusters_list=[2]#[2,6,12,20]
output_sizes=[400]#,200,400,800,1600]#[10,100,600,1200]

group_dim_reduce=False
#reruns= np.linspace(1,16,16)

cluster_methods=['ward']
cross_cluster=True

affinity_thresh= 0.0
ism_gsm_stability=[]
#ind_clust_stab_mat=[]
#ind_clust_stab_summary=[[1, 2, 3, 4, 5]]
run=True


analysis_ID='testing_ism_gsms4'
reruns= 1


run_PyBASC(dataset_bootstrap_list,timeseries_bootstrap_list, similarity_metric_list, cluster_methods, 
         blocklength_list, n_clusters_list, output_sizes, subject_file_list, roi_mask_file, proc_mem,
         cross_cluster, roi2_mask_file, affinity_thresh, run, home, reruns, group_dim_reduce, analysis_ID)
