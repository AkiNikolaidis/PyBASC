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
proc_mem= [30,220]


subject_file_list = [
		'/home/ec2-user/refdata/0025427_gsr-1_scrub-0.nii.gz',
		'/home/ec2-user/refdata/0025428_gsr-1_scrub-0.nii.gz',
		'/home/ec2-user/refdata/0025429_gsr-1_scrub-0.nii.gz',
		'/home/ec2-user/refdata/0025430_gsr-1_scrub-0.nii.gz',
		'/home/ec2-user/refdata/0025431_gsr-1_scrub-0.nii.gz',
		'/home/ec2-user/refdata/0025432_gsr-1_scrub-0.nii.gz',
		'/home/ec2-user/refdata/0025433_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025434_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025435_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025436_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025437_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025438_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025439_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025440_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025441_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025442_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025443_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025444_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025445_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025446_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025447_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025448_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025449_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025450_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025451_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025452_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025453_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025454_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025455_gsr-1_scrub-0.nii.gz',
        '/home/ec2-user/refdata/0025456_gsr-1_scrub-0.nii.gz']



#roi_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Full_BG_Sim_3mm.nii.gz'
#roi2_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Yeo7_3mmMasks/Yeo_2_3mm.nii.gz'

roi_mask_file='masks/Full_BG_Sim_3mm.nii.gz'


roi2_mask_file='masks/Full_BG_Sim_3mm.nii.gz' #'masks/Yeo7_3mmMasks/Yeo_2_3mm.nii.gz' #


dataset_bootstrap_list=[100,200,400,800,1600]#[100,200,400,800,1600]#,10,30,100]
timeseries_bootstrap_list=[100,200,400,800,1600]#[100,200,400,800,1600]#,10,30,100]
similarity_metric_list=['correlation'] #['correlation','euclidean','cityblock', 'cosine']
blocklength_list=[1]
n_clusters_list=[4]#[2,6,12,20]
output_sizes=[800]#,200,400,800,1600]#[10,100,600,1200]

group_dim_reduce=False
#reruns= np.linspace(1,16,16)
cross_cluster=False
affinity_thresh= 0.0
ism_gsm_stability=[]
#ind_clust_stab_mat=[]
#ind_clust_stab_summary=[[1, 2, 3, 4, 5]]
run=True


analysis_ID='BootstrapTest_Rep'
reruns= 30


run_PyBASC(dataset_bootstrap_list,timeseries_bootstrap_list, similarity_metric_list,
         blocklength_list, n_clusters_list, output_sizes, subject_file_list, roi_mask_file, proc_mem,
         cross_cluster, roi2_mask_file, affinity_thresh, run, home, reruns, group_dim_reduce, analysis_ID)