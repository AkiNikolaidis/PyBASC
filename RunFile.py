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
from os.path import expanduser


home = expanduser("~")
proc_mem= [2,4]

from BASC.basc_workflow_runner import run_basc_workflow
import BASC.utils


subject_file_list= [home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/BASC/sample_data/sub3/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                    home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz']

#'/Users/aki.nikolaidis/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz'

roi_mask_file= home + '/git_repo/BASC/masks/LC_Quarter_Res.nii.gz'
dataset_bootstraps=50
timeseries_bootstraps=10
n_clusters=7
output_size=10
bootstrap_list=list(range(0,dataset_bootstraps))
cross_cluster=True
roi2_mask_file= home + '/git_repo/BASC/masks/RC_Quarter_Res.nii.gz'
affinity_threshold= [0.5] * len(subject_file_list)
out_dir= home + '/BASC_outputs/singlescripttest'
run=True



basc_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)

