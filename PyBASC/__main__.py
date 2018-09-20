#import PyBASC.__init__
#import PyBASC.utils
import PyBASC
from PyBASC import *
import os
import gc
import numpy as np
import scipy.stats
from os.path import expanduser
import pkg_resources
import yaml
import sys


#from basc_workflow_runner import run_basc_workflow
#from basc import save_igcm_nifti, create_group_cluster_maps, ism_nifti, gsm_nifti

#group_dim_reduce=False

def main(config):
    
    #import pdb;pdb.set_trace()
    #print(config)
    #import pdb;pdb.set_trace()

    print(config)

    dataset_bootstrap_list=config['dataset_bootstrap_list']
    timeseries_bootstrap_list=config['timeseries_bootstrap_list']
    similarity_metric_list=config['similarity_metric_list']
    blocklength_list=config['blocklength_list']
    n_clusters_list=config['n_clusters_list']
    output_sizes=config['output_sizes']
    subject_file_list=config['subject_file_list']
    roi_mask_file=config['roi_mask_file']
    proc_mem=config['proc_mem']
    cross_cluster=config['cross_cluster']
    roi2_mask_file=config['roi2_mask_file']
    affinity_thresh=config['affinity_thresh']
    run=config['run']
    home=config['home']
    reruns=config['reruns']
    group_dim_reduce=config['group_dim_reduce']
    analysis_ID=config['analysis_ID']
    
    #import pdb;pbd.set_trace()
    
    run_PyBASC(dataset_bootstrap_list,timeseries_bootstrap_list, similarity_metric_list, 
         blocklength_list, n_clusters_list, output_sizes, subject_file_list, roi_mask_file, proc_mem,
         cross_cluster, roi2_mask_file, affinity_thresh, run, home, reruns, group_dim_reduce, analysis_ID)


def run_PyBASC(dataset_bootstrap_list,timeseries_bootstrap_list, similarity_metric_list, 
         blocklength_list, n_clusters_list, output_sizes, subject_file_list, roi_mask_file, proc_mem,
         cross_cluster, roi2_mask_file, affinity_thresh, run, home, reruns, group_dim_reduce, analysis_ID):



    ism_gsm_stability=[]
    ind_clust_stab_summary=[[1, 2, 3, 4, 5]]
    rerun_list= np.linspace(1,reruns,reruns)
    roi_mask_file=pkg_resources.resource_filename('PyBASC', roi_mask_file)
    roi2_mask_file=pkg_resources.resource_filename('PyBASC', roi2_mask_file)
    affinity_threshold= [affinity_thresh] * len(subject_file_list)

    for rerun in rerun_list:    
        #roi_mask_file=
        #import pdb; pdb.set_trace()
        
        randseed=np.random.randint(0,10000)
        np.random.seed(randseed)

        
        for (dataset_bootstraps, timeseries_bootstraps) in zip(dataset_bootstrap_list,timeseries_bootstrap_list):
            if dataset_bootstraps==0:
                bootstrap_list=[0]
            else:
                bootstrap_list= list(np.ones(dataset_bootstraps, dtype=int)*dataset_bootstraps)
            
            
            for similarity_metric in similarity_metric_list:
                for blocklength in blocklength_list:
                        for n_clusters in n_clusters_list:
                            for output_size in output_sizes:
                                #import pdb; pdb.set_trace()
                                #import pdb; pdb.set_trace()
                                
                                out_dir= home + '/PyBASC_outputs/' + analysis_ID + '/Run_' + str(int(rerun)) + '_' + str(output_size) + '_' + str(similarity_metric) + '_' + str(n_clusters) + '_clusters_' +str(timeseries_bootstraps) +'_IndBS_' + str(blocklength) + '_block' + similarity_metric
                                #out_dir= '/Users/aki.nikolaidis/PyBASC_outputs/Testing_Ward/dim_' + str(output_size) + '_' + str(similarity_metric) + '_' + str(n_clusters) + '_clusters_' +str(timeseries_bootstraps) +'_IndBS_' + str(blocklength) + '_block' + similarity_metric
                                PyBASC_test=run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, similarity_metric, group_dim_reduce=group_dim_reduce, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, blocklength=blocklength, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)
    
        #                        del PyBASC_test
        #                        gc.collect()
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
                            
                            print('saving files: ism_gsm_stability')
                            ism_gsm_stability_file=os.path.join(out_dir, 'ism_gsm_stability_'+ str(n_clusters)+ '.npy')
                            np.save(ism_gsm_stability_file, ism_gsm_stability)
                            ism_gsm_stability=[]
                    
                        print('saving files: ind_clust_stab_summary')
                        ind_clust_stab_summary_file=os.path.join(out_dir, 'ind_clust_stab_summary.npy')
                        np.save(ind_clust_stab_summary_file, ind_clust_stab_summary)