#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:24:25 2017

@author: aki.nikolaidis
"""


#def clusternum_stability():
import basc
import utils
import os
import numpy as np
basc_output_list = ['/Users/aki.nikolaidis/BASC_outputs/ClusterComparisons/testing2/workflow_output',
                   '/Users/aki.nikolaidis/BASC_outputs/ClusterComparisons/testing4/workflow_output',
                   '/Users/aki.nikolaidis/BASC_outputs/ClusterComparisons/testing6/workflow_output',
                   '/Users/aki.nikolaidis/BASC_outputs/ClusterComparisons/testing7/workflow_output'
                   ]

ism_gsm_stability=[]
for k in len(basc_output_list):
    gsmfile= basc_output_list[k] + '/gsm/group_stability_matrix.npy'
    gsm=np.load(gsmfile)
    ismpath=basc_output_list[k] + '/basc_workflow_runner/basc/individual_stability_matrices/mapflow'
    ([dirpath, dirnames,filenames])= os.walk(ismpath)
    for (dirpath, dirnames,filenames) in os.walk(ismpath):
        print(dirnames)
        ismfile= dirnames + '/individual_stability_matrix.npy'
#        ismfile= basc_output_list[k] + '/basc_workflow_runner/basc/individual_stability_matrices/mapflow/_individual_stability_matrices' +str(i)+ '/individual_stability_matrix.npy'

#        ism=np.load(ismfile)
#        ism_gsm_stability[i,k].append()= utils.compare_stability_matrices()
#        
#        
#        /Users/aki.nikolaidis/BASC_outputs/ClusterComparisons/testing2/workflow_output/basc_workflow_runner/basc/individual_stability_matrices/mapflow/