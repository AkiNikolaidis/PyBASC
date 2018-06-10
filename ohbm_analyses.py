#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:31:56 2018

@author: aki.nikolaidis
"""

### Analysis of OHBM 9 Corticostriatal Networks
#For both Reference and Replication Datasets:
#* Loop over the each cluster number [2,4,…20]-----------------------------DONE
#    * Loop over each of the 9 networks------------------------------------DONE
#    * Load the cluster labels and put into a matrix-----------------------DONE
#* After building the cluster label matrix
#* Initialize a cluster distance matrix 90x90------------------------------DONE
#* Then loop over each column in the matrix, (col_idx_1)-------------------DONE
#    * Loop over each column in the matrix (col_idx_2)---------------------DONE
#        * Compute the adjusted rand index between the cluster assignments.DONE
#        * Put computed rand index into a cluster distance matrix----------DONE

#* Calculate spatial correlation between Reference and Replication Dataset,----------------------------------------------DONE
#       for each of the 10, 9x9 matrices.
#    * Clusters2corr= np.corrcoef(RefClusterSimilarity(0:9,0:9).ravel(), RepClusterSimilarity(0:9,0:9).ravel())[0][1]----DONE
#    * Clusters4=cols10:19,rows10:19
#    * Etc etc
#    * Clusters 20= cols80:90, rows80:89
#* Calculate spatial correlation between reference and replication for the full cluster distance matrix.----------------DONE


import numpy as np
import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score

#/pybasc-output//40min-data-output/40mindata/40MinData100_GS_/home/ec2-user/git_repo/PyBASC/masks/Yeo7_3mmMasks/Yeo_1_3mm.nii.gz/dim_800_correlation_10_clusters_100_IndBS_1_blockcorrelation/workflow_output/basc_workflow_runner/basc/individual_stability_matrices/mapflow/_individual_stability_matrices0/?region=us-east-2&tab=overview
all_labels=pd.DataFrame()



clusterg_path='/clusters_G/clusters_G.npy'
data_dir='/data2/Projects/BASC/HNU_SSI/PyBASC_outputs/AWS_outputs/40min-data-output'
ref_dir='/data2/Projects/BASC/HNU_SSI/PyBASC_outputs/AWS_outputs/ref-data-output'
subjects= np.linspace(0,29,30)
clusternum_list= ['2','4','6','8','10','12','14','16','18','20']

network_list=['Full_BG_Sim_3mm.nii.gz', 'Yeo_1_3mm.nii.gz','Yeo_2_3mm.nii.gz','Yeo_3_3mm.nii.gz','Yeo_4_3mm.nii.gz','Yeo_5_3mm.nii.gz','Yeo_6_3mm.nii.gz','Yeo_7_3mm.nii.gz', 'cerebellum_3mm.nii.gz']

for clusternum in clusternum_list:
    for network in network_list:
        current_path= data_dir + '/' + network + '/dim_800_correlation_'+ clusternum + '_clusters_100_IndBS_1_blockcorrelation/workflow_output'
        group_labels_path= current_path + clusterg_path
        #print(group_labels_path)
        #clust_label_temp=np.load(workflowpath to clusters_g.npy)
        #import pdb;pdb.set_trace()
        group_labels=np.load(group_labels_path)
        print(group_labels_path)
        new_column_name= clusternum+ '_clusters_' + network
        all_labels[new_column_name] = group_labels
        
        #add a column to a dataframe, where each column of the dataframe is labeled by the clusternum and networklist.
        
        
        #for subject in subjects:
        #    subject=str(int(subject))
            #print(subject)
        #    ismdir=current_path + 'basc_workflow_runner/basc/individual_stability_matrices/mapflow/_individual_stability_matrices'+ subject
            #ism=np.load(ismdir)
            

label_sim_matrix=pd.DataFrame(columns=list(all_labels), index=list(all_labels))

for column in all_labels:
    column1=column
    for column in all_labels:
        column2=column
        #import pdb; pdb.set_trace()
        #print(column1, column2)
        col1loc=all_labels.columns.get_loc(column1)
        col2loc=all_labels.columns.get_loc(column2)
        #import pdb; pdb.set_trace()
        label_sim_matrix[column1][column2]= adjusted_rand_score(all_labels[column1],all_labels[column2])
        
#%%


#np.save('./all_labels_40Min',all_labels)
#np.save('./label_sim_matrix_40Min',label_sim_matrix)
all_labels_40Min=all_labels
label_sim_matrix_40Min=label_sim_matrix
label_sim_matrix_40Min=label_sim_matrix_40Min.astype(float)

label_sim_matrix_Ref=label_sim_matrix
label_sim_matrix_Ref=label_sim_matrix_Ref.astype(float)


a=0
b=9
clusters_corr_all=[]
clusternum_all=[]
for clusternum in clusternum_list:
    import pdb; pdb.set_trace()
    #plt.imshow(label_sim_matrix_Ref[label_sim_matrix_Ref.columns[a:b]][a:b].values);plt.show()
    #plt.imshow(label_sim_matrix_Ref[label_sim_matrix_40Min.columns[a:b]][a:b].values);plt.show()
    print(a)
    print(b)
    rep_cluster=label_sim_matrix_40Min[label_sim_matrix_40Min.columns[a:b]][a:b].values.ravel()
    ref_cluster=label_sim_matrix_Ref[label_sim_matrix_Ref.columns[a:b]][a:b].values.ravel()
    clusters_corr=np.corrcoef(rep_cluster,ref_cluster)[0][1]
    clusters_corr_all.append(clusters_corr)
    a=a+10
    b=b+10
    
corrdata_perclusternum=pd.DataFrame(
        {'clusternum':clusternum_list,
                       'Ref-Rep-Correlation':clusters_corr_all})

global_corrdata=np.corrcoef(label_sim_matrix_40Min.values.ravel(),label_sim_matrix_Ref.values.ravel())[0][1]



#%%
def ism_nifti(roi_mask_file, n_clusters, out_dir): #NEED TO CHANGE THIS SCRIPT TO:
    #APPLY GROUP LEVEL CLUSTER LABELS TO INDIVIDUAL LEVELS
    #
    #EXTRACT VOXELWISE STABILITY INFO FOR THAT CLUSTER ACROSS ALL PARTICIPANTS
    #USE KMASK TO CREATE THAT CLUSTER INFORMATION
    #
    
    #Loop over all ISMs, 
#        load ISM, 
#        loop over all clusters, 
#            perform calculation, 
#            add calculation to running tabs.
#    #Calculate mean at each voxel and CV- 
#    output a voxelwise mean stability and CV Nifti map for each cluster.
    
    
    """
    Calculate the individual level stability and instability maps for each of the group clusters.
    Create Nifti files for each individual cluster's stability map
    
    
    Parameters
    ----------
        roi_mask_file: the mask of the region to calculate stability for.
        n_clusters: the number of clusters calculated
        out_dir: the directory to output the saved nifti images
    

    Returns
    -------
    Creates NIFTI files for all the ism cluster stability maps
    
    """
    import utils
    import basc
    import numpy as np
    import os
    from pathlib import Path
    
    #*ACTION - FIGURE OUT IF CAN BE ADDED TO BASC WORKFLOW, OR DIFFERENT WORKFLOW?
    
    #Individual subject ISM to NIFTI and individual
    #Inputs Subject ISM, ROIFile, 


    
    
    #for i in range(nSubjects):
    ismdir=out_dir + '/workflow_output/basc_workflow_runner/basc/individual_stability_matrices/mapflow/'
    os.chdir(ismdir)
    subdirs_all = [x[1] for x in os.walk(ismdir)]                                                                            
    subdirs=subdirs_all[0]
    roi_mask_nparray = nb.load(roi_mask_file).get_data().astype('float32').astype('bool')

    for subdir in subdirs:
        os.chdir(ismdir + subdir)
        
        ind_cluster_stability_file = os.path.join(os.getcwd(), 'ind_cluster_stability.npy')
        ind_cluster_INSTABILITY_file = os.path.join(os.getcwd(), 'ind_cluster_INSTABILITY.npy')
        ind_cluster_stability_Diff_file = os.path.join(os.getcwd(), 'ind_cluster_stability_Diff.npy')
        ism_cluster_voxel_scores_file = os.path.join(os.getcwd(), 'ism_cluster_voxel_scores.npy')
        
        end_file = Path(ism_cluster_voxel_scores_file)
        
        if end_file.exists():
            
            return
        else:
            
            ism=np.load(ismdir + subdir + '/individual_stability_matrix.npy')
            clusters_ism = utils.cluster_timeseries(ism, roi_mask_nparray, n_clusters, similarity_metric = 'correlation', affinity_threshold=0.0, cluster_method='ward')
            clusters_ism = clusters_ism+1
            niftifilename = ismdir + subdir +'/ism_clust.nii.gz'
            clusters_ism_file = ismdir + subdir +'/clusters_ism.npy'
            #Saving Individual Level Cluster Solution
            ndarray_to_vol(clusters_ism, roi_mask_file, roi_mask_file, niftifilename)
            np.save(clusters_ism_file, clusters_ism)
            
            
            cluster_ids = np.unique(clusters_ism)
            nClusters = cluster_ids.shape[0]
            nVoxels = clusters_ism.shape[0]
            ism_cluster_voxel_scores = np.zeros((nClusters, nVoxels))
            k_mask=np.zeros((nVoxels, nVoxels))
            ism_cluster_voxel_scores[:,:], k_mask[:,:] = utils.cluster_matrix_average(ism, clusters_ism)
            ism_cluster_voxel_scores=ism_cluster_voxel_scores.astype("uint8")
            
            ind_cluster_stability=[]
            ind_cluster_INSTABILITY=[]
            ind_cluster_stability_Diff=[]
            
    #        ind_cluster_stability_file = os.path.join(os.getcwd(), 'ind_cluster_stability.npy')
    #        ind_cluster_INSTABILITY_file = os.path.join(os.getcwd(), 'ind_cluster_INSTABILITY.npy')
    #        ind_cluster_stability_Diff_file = os.path.join(os.getcwd(), 'ind_cluster_stability_Diff.npy')
    #        ism_cluster_voxel_scores_file = os.path.join(os.getcwd(), 'ism_cluster_voxel_scores.npy')
    #        
            os.chdir(ismdir + '/' + subdir)
            

