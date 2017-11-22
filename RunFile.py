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
import gc
import numpy as np
import scipy.stats
from os.path import expanduser
from basc_workflow_runner import run_basc_workflow
from basc import save_igcm_nifti, create_group_cluster_maps, ism_nifti, gsm_nifti

home = expanduser("~")
proc_mem= [3,6]




#subject_file_list= [home + '/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
#                    home + '/git_repo/PyBASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
#                    home + '/git_repo/PyBASC/sample_data/sub3/Func_Quarter_Res.nii.gz',
#                    home + '/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
#                    home + '/git_repo/PyBASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
#                    home + '/git_repo/PyBASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
#                    home + '/git_repo/PyBASC/sample_data/sub2/Func_Quarter_Res.nii.gz']
#

subject_file_list = ['/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_100_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_100_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_100_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_171_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_171_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_171_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_200_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_200_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_200_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_285_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_285_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_285_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_300_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_300_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_300_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_42_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_42_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_42_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_50_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_50_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_50_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_85_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_85_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/1400_Data/TimeTest/1400_85_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_100_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_100_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_100_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_186_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_186_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_186_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_200_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_200_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_200_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_300_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_300_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_300_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_372_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_372_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_372_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_50_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_50_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_50_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_600_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_600_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_600_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_620_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_620_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_620_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_852_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_852_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_852_A00040915.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_93_A00029979.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_93_A00031217.nii.gz',
'/Users/aki.nikolaidis/Desktop/NKI_Dev/645_Data/TimeTest/645_93_A00040915.nii.gz']

roi_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Yeo7_3mmMasks/BilateralStriatumThalamus_3mm.nii.gz'
roi2_mask_file='/Users/aki.nikolaidis/git_repo/PyBASC/masks/Yeo7_3mmMasks/Yeo_All_7_3mm.nii.gz'





##
#roi_mask_file= home + '/git_repo/PyBASC/masks/LC_Quarter_Res.nii.gz'
#roi2_mask_file= home + '/git_repo/PyBASC/masks/RC_Quarter_Res.nii.gz'

dataset_bootstraps=2
timeseries_bootstrap_list=[1,10,100,200]
n_clusters_list=[2,6,12,20]
output_sizes=[10,100,600,1200]
blocklength_list=[0.5,1,2]
similarity_metric_list= ['correlation','euclidean','cityblock', 'cosine']

bootstrap_list=list(range(0,dataset_bootstraps))
cross_cluster=True
affinity_threshold= [0.0] * len(subject_file_list)

ism_gsm_stability=[]
ind_clust_stab_mat=[]
ind_clust_stab_summary=[[1, 2, 3, 4, 5]]
run=True

for timeseries_bootstraps in timeseries_bootstrap_list:
    for similarity_metric in similarity_metric_list:
        for blocklength in blocklength_list:
                for n_clusters in n_clusters_list:
                    for output_size in output_sizes:
                        #import pdb; pdb.set_trace()
                        out_dir= home + '/PyBASC_outputs/ISM_Testing_affinity00/dim_' + str(output_size) + '_' + str(similarity_metric) + '_' + str(n_clusters) + '_clusters_' +str(timeseries_bootstraps) +'_IndBS_' + str(blocklength) + '_block' + similarity_metric
                        PyBASC_test=run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, similarity_metric, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, blocklength=blocklength, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)
                        del PyBASC_test
                        gc.collect()
                        #import pdb; pdb.set_trace()
                        ism_gsm_stability.append(np.load(out_dir + '/workflow_output/ism_gsm_corr_file/ism_gsm_corr.npy'))
                        ind_clust_stab_mat = np.load(out_dir + '/workflow_output/ind_group_cluster_stability_set/ind_group_cluster_stability_set.npy')
                        ind_clust_stab_summary=np.concatenate((ind_clust_stab_summary, np.array([[n_clusters, output_size, ind_clust_stab_mat.mean(), scipy.stats.variation(ind_clust_stab_mat).mean(), (ind_clust_stab_mat.mean() - scipy.stats.variation(ind_clust_stab_mat).mean())]])))
                        
                        #Run Group ClusterMaps
                        gsm_file = out_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/group_stability_matrix.npy'
                        clusters_G_file = out_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/clusters_G.npy'
                        os.chdir(out_dir +'/workflow_output/basc_workflow_runner/basc/join_group_stability/')
                        create_group_cluster_maps(gsm_file,clusters_G_file,roi_mask_file)
                        #Run IGCM on all individual subjects
                        clustvoxscoredir=out_dir + '/workflow_output/basc_workflow_runner/basc/individual_group_clustered_maps/mapflow/'
                        clusters_G_file= out_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/clusters_G.npy'
                        #clusters_G = np.load(clusters_G_file)
                        
                        ism_nifti(roi_mask_file, n_clusters, out_dir)
                        gsm_nifti(roi_mask_file, n_clusters, out_dir)
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
                    
                    print('saving files: ism_gsm_stability')
                    ism_gsm_stability_file=os.path.join(out_dir, 'ism_gsm_stability_'+ str(n_clusters)+ '.npy')
                    np.save(ism_gsm_stability_file, ism_gsm_stability)
                    ism_gsm_stability=[]
            
                print('saving files: ind_clust_stab_summary')
                ind_clust_stab_summary_file=os.path.join(out_dir, 'ind_clust_stab_summary.npy')
                np.save(ind_clust_stab_summary_file, ind_clust_stab_summary)
    
    #PSEUDO CODE- FOR THE ANALYSIS WITH THE HIGHEST MEAN AND LOWEST CV
    # RUN POST ANALYSIS- INCLUDING FOR LOOPING OVER ALL SUBJECTS FOLDERS AND CREATING NIFTI FILES FOR EVERY CLUSTER
    
    
    
        
#clustervoxscores=np.load(outdir,)
