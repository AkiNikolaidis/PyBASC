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
proc_mem= [3,8]


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


roi_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Full_BG_Sim_3mm.nii.gz'
roi2_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Yeo7_3mmMasks/Yeo_2_3mm.nii.gz'

dataset_bootstraps_list=[30]#,10,30,100]
timeseries_bootstrap_list=[30]#,10,30,100]
similarity_metric_list=['correlation'] #['correlation','euclidean','cityblock', 'cosine']
blocklength_list=[1]#[0.5,1,2]
n_clusters_list=[2]#[2,6,12,20]
output_sizes=[100,200,400,800,1600]#[10,100,600,1200]

group_dim_reduce=False
cross_cluster=True
affinity_threshold= [0.0] * len(subject_file_list)

ism_gsm_stability=[]
ind_clust_stab_mat=[]
ind_clust_stab_summary=[[1, 2, 3, 4, 5]]
run=True

reruns= np.linspace(1,16,16)


for rerun in reruns:
    randseed=np.random.randint(0,10000)
    np.random.seed(randseed)
    for dataset_bootstraps in dataset_bootstraps_list:
        #bootstrap_list=list(range(0, dataset_bootstraps))
        bootstrap_list= list(np.ones(dataset_bootstraps, dtype=int)*dataset_bootstraps)
    
    
        for timeseries_bootstraps in timeseries_bootstrap_list:
            for similarity_metric in similarity_metric_list:
                for blocklength in blocklength_list:
                        for n_clusters in n_clusters_list:
                            for output_size in output_sizes:
                                #import pdb; pdb.set_trace()
                                #out_dir= '/home/ec2-user/PyBASC_outputs/SimTesting/dim_' + str(output_size) + '_' + str(similarity_metric) + '_' + str(n_clusters) + '_clusters_' +str(timeseries_bootstraps) +'_IndBS_' + str(blocklength) + '_block' + similarity_metric
                                out_dir= '/Users/aki.nikolaidis/PyBASC_outputs/GroupDimReduce_AccTest/NewGroupDimOff_' + str(int(rerun)) + '_' + str(dataset_bootstraps) + 'GS' +'/dim_' + str(output_size) + '_' + str(similarity_metric) + '_' + str(n_clusters) + '_clusters_' +str(timeseries_bootstraps) +'_IndBS_' + str(blocklength) + '_block' + similarity_metric
                                #import pdb;pdb.set_trace()
                                PyBASC_test=run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, similarity_metric, group_dim_reduce=group_dim_reduce, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, blocklength=blocklength, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)
                                
                                #import pdb; pdb.set_trace()
#                                PyBASC_pipeline=PyBASC_test[0]
#                                PyBASC_pipeline.write_graph(graph2use='exec', dotfilename='./graph_exec.dot', simple_form=False)
#                                #import pdb; pdb.set_trace()
#                                from IPython.display import Image
#                                Image(filename="graph_exec_detailed.dot.png")
                                #import pdb; pdb.set_trace()
        #                        del PyBASC_test
        #                        gc.collect()
                                #import pdb; pdb.set_trace()
#                                ism_gsm_stability.append(np.load(out_dir + '/workflow_output/ism_gsm_corr_file/ism_gsm_corr.npy'))
#                                ind_clust_stab_mat = np.load(out_dir + '/workflow_output/ind_group_cluster_stability_set/ind_group_cluster_stability_set.npy')
#                                ind_clust_stab_summary=np.concatenate((ind_clust_stab_summary, np.array([[n_clusters, output_size, ind_clust_stab_mat.mean(), scipy.stats.variation(ind_clust_stab_mat).mean(), (ind_clust_stab_mat.mean() - scipy.stats.variation(ind_clust_stab_mat).mean())]])))
#                                
#                                #Run Group ClusterMaps
#                                gsm_file = out_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/group_stability_matrix.npy'
#                                clusters_G_file = out_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/clusters_G.npy'
#                                os.chdir(out_dir +'/workflow_output/basc_workflow_runner/basc/join_group_stability/')
#                                create_group_cluster_maps(gsm_file,clusters_G_file,roi_mask_file)
#                                #Run IGCM on all individual subjects
#                                clustvoxscoredir=out_dir + '/workflow_output/basc_workflow_runner/basc/individual_group_clustered_maps/mapflow/'
#                                clusters_G_file= out_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/clusters_G.npy'
#                                #clusters_G = np.load(clusters_G_file)
                                
                                #ism_nifti(roi_mask_file, n_clusters, out_dir)
                                #gsm_nifti(roi_mask_file, n_clusters, out_dir)
                                
                                
                                
                                
                        #        subdirs_all = [x[1] for x in os.walk(clustvoxscoredir)]                                                                            
                        #        subdirs=subdirs_all[0]
                        #        for subdir in subdirs:
                        #        
                    #            #import pdb; pdb.set_trace()
                    #            clustvoxscorefile=clustvoxscoredir + subdir+ '/cluster_voxel_scores.npy'
                    #            #clustvoxscores=np.load(clustvoxscorefile)
                    #            os.chdir(clustvoxscoredir + '/' + subdir)
                    #            save_igcm_nifti(clustvoxscorefile,clusters_G_file, roi_mask_file)
                    #           
                    
                                
                            
#                            print('saving files: ism_gsm_stability')
#                            ism_gsm_stability_file=os.path.join(out_dir, 'ism_gsm_stability_'+ str(n_clusters)+ '.npy')
#                            np.save(ism_gsm_stability_file, ism_gsm_stability)
#                            ism_gsm_stability=[]
#                    
#                        print('saving files: ind_clust_stab_summary')
#                        ind_clust_stab_summary_file=os.path.join(out_dir, 'ind_clust_stab_summary.npy')
#                        np.save(ind_clust_stab_summary_file, ind_clust_stab_summary)
#            
        #PSEUDO CODE- FOR THE ANALYSIS WITH THE HIGHEST MEAN AND LOWEST CV
        # RUN POST ANALYSIS- INCLUDING FOR LOOPING OVER ALL SUBJECTS FOLDERS AND CREATING NIFTI FILES FOR EVERY CLUSTER
        
    
os.system('say "Hello, your analysis has completed. Please take a look at the output files, have a nice day, and dont forget to tip your server"')    
        
#clustervoxscores=np.load(outdir,)
