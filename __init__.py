#import BASC.utils #as utils

from utils import timeseries_bootstrap, \
                  standard_bootstrap, \
                  cluster_timeseries, \
                  cross_cluster_timeseries, \
                  adjacency_matrix, \
                  cluster_matrix_average, \
                  individual_stability_matrix, \
                  expand_ism, \
                  compare_stability_matrices, \
                  data_compression

#import BASC.basc

from basc import create_basc, \
                 nifti_individual_stability, \
                 map_group_stability, \
                 join_group_stability, \
                 ndarray_to_vol, \
                 individual_group_clustered_maps
            
#from BASC.basc_workflow_runner import run_basc_workflow as run_basc_workflow

import RunFile as RunFile

              
            
__all__ = [       'timeseries_bootstrap', \
                  'standard_bootstrap', \
                  'cluster_timeseries', \
                  'cross_cluster_timeseries', \
                  'adjacency_matrix', \
                  'cluster_matrix_average', \
                  'individual_stability_matrix', \
                  'expand_ism', \
                  'compare_stability_matrices', \
                  'data_compression', \
                  
                  'create_basc', \
                  'nifti_individual_stability', \
                  'map_group_stability', \
                  'join_group_stability', \
                  'ndarray_to_vol', \
                  'individual_group_clustered_maps'
                 
                 ]
        

#        
#        'create_basc', \
#           'nifti_individual_stability', \
#           'map_group_stability', \
#           'join_group_stability', \
#           'individual_group_clustered_maps'
#           'timeseries_bootstrap', \
#           'standard_bootstrap', \
#           'cluster_timeseries', \
#           'cross_cluster_timeseries', \
#           'adjacency_matrix', \
#           'cluster_matrix_average', \
#           'individual_stability_matrix', \
#           'data_compression', \
#           'run_basc_workflow']

#adding in a test file