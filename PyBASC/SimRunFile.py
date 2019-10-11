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

import os
import gc
import numpy as np
import nibabel as nb

import scipy.stats
from os.path import expanduser
from basc_workflow_runner import run_basc_workflow
from basc import save_igcm_nifti, create_group_cluster_maps, ism_nifti, gsm_nifti
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:32:13 2019

@author: aki.nikolaidis
"""


with open('/Users/aki.nikolaidis/git_repo/PyBASC/PyBASC/testing_256.yaml', "r") as f:
        config = yaml.load(f)

if 'home' in config:
    home = os.path.abspath(config['home'])
    os.chdir(home)
else:
    home = os.getcwd()
    
analysis_id = config['analysis_ID']
run = config['run']
proc_mem = config['proc_mem']
path = os.path.dirname(PyBASC.__file__)

random_seed = config['random_seed']

subject_file_list = [
    os.path.abspath(s.replace('$PYBASC', path))
    for s in config['subject_file_list']
]

reruns = config.get('reruns', 1)
dataset_bootstraps_list = config['dataset_bootstrap_list']
timeseries_bootstraps_list = config['timeseries_bootstrap_list']
similarity_metric_list = config['similarity_metric_list']
cluster_method_list = config['cluster_methods']
blocklength_list = config['blocklength_list']
n_clusters_list = config['n_clusters_list']
output_size_list = config['output_sizes']
affinity_threshold_list = config['affinity_threshold_list']
roi_mask_file = config['roi_mask_file']
cross_cluster = config.get('cross_cluster', False)
cross_cluster_mask_file = config.get('cross_cluster_mask_file', None)
group_dim_reduce = config.get('group_dim_reduce', False)

roi_mask_file = os.path.abspath(roi_mask_file.replace('$PYBASC', path))
if cross_cluster_mask_file:
    cross_cluster_mask_file = \
        os.path.abspath(cross_cluster_mask_file.replace('$PYBASC', path))



run_PyBASC(
            dataset_bootstrap_list=dataset_bootstraps_list,
            timeseries_bootstrap_list=timeseries_bootstraps_list,
            similarity_metric_list=similarity_metric_list,
            cluster_methods=cluster_method_list,
            blocklength_list=blocklength_list,
            n_clusters_list=n_clusters_list,
            output_sizes=output_size_list,
            subject_file_list=subject_file_list,
            roi_mask_file=roi_mask_file,
            proc_mem=proc_mem,
            cross_cluster=cross_cluster,
            cross_cluster_mask_file=cross_cluster_mask_file,
            affinity_threshold_list=affinity_threshold_list,
            run=run,
            home=home,
            reruns=reruns,
            group_dim_reduce=group_dim_reduce,
            analysis_ID=analysis_id,
            random_seed=random_seed
        )

