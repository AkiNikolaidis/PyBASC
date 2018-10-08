import os
import sys

import numpy as np
import pkg_resources
import scipy.stats
import yaml
import scipy.sparse
import warnings
import PyBASC
from PyBASC import create_group_cluster_maps, run_basc_workflow

warnings.filterwarnings("ignore")

def main_args():

    import yaml
    import sys
    import argparse

    from PyBASC import main

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config file', type=argparse.FileType('r'))
    args = parser.parse_args()

    config = yaml.load(args.config)

    main(config)



def main(config):

    if 'home' in config:
        home = os.path.abspath(config['home'])
        os.chdir(home)
    else:
        home = os.getcwd()
    
    analysis_ID = config['analysis_ID']
    run = config['run']
    proc_mem = config['proc_mem']
    path = os.path.dirname(PyBASC.__file__)
    
    subject_file_list = [
        os.path.abspath(s.replace('$PYBASC',path))
        for s in config['subject_file_list']
    ]


    reruns = config['reruns']
    dataset_bootstrap_list = config['dataset_bootstrap_list']
    timeseries_bootstrap_list = config['timeseries_bootstrap_list']
    similarity_metric_list = config['similarity_metric_list']
    cluster_methods = config['cluster_methods']
    blocklength_list = config['blocklength_list']
    n_clusters_list = config['n_clusters_list']
    output_sizes = config['output_sizes']
    roi_mask_file = config['roi_mask_file']
    cross_cluster = config['cross_cluster']
    cross_cluster_mask_file = config['cross_cluster_mask_file']
    affinity_thresh = config['affinity_thresh']
    group_dim_reduce = config['group_dim_reduce']




    run_PyBASC(dataset_bootstrap_list, timeseries_bootstrap_list, similarity_metric_list, cluster_methods,
               blocklength_list, n_clusters_list, output_sizes, subject_file_list, roi_mask_file, proc_mem,
               cross_cluster, cross_cluster_mask_file, affinity_thresh, run, home, reruns, group_dim_reduce, analysis_ID)


def run_PyBASC(
    dataset_bootstrap_list, timeseries_bootstrap_list, similarity_metric_list, cluster_methods,
    blocklength_list, n_clusters_list, output_sizes, subject_file_list, roi_mask_file, proc_mem,
    cross_cluster, cross_cluster_mask_file, affinity_thresh, run, home, reruns, group_dim_reduce, analysis_ID,
    random_seed=None
):

    ism_gsm_stability = []
    ind_clust_stab_summary = [[1, 2, 3, 4, 5]]
    rerun_list = np.arange(reruns)
    affinity_threshold = [affinity_thresh] * len(subject_file_list)

    roi_mask_file = pkg_resources.resource_filename('PyBASC', roi_mask_file)
    cross_cluster_mask_file = pkg_resources.resource_filename('PyBASC', cross_cluster_mask_file)

    out_dir = os.path.join(
        home,
        'PyBASC_outputs',
        '{analysis}',
        '_'.join([
            'Run_{rerun}',
            '{output_size}',
            '{similarity_metric}',
            '{cluster_method}',
            'clusters-{n_clusters}',
            'IndBS-{timeseries_bootstraps}',
            'block-{blocklength}'
        ])
    )

    

    for rerun in rerun_list:
        randseed=np.random.randint(0,10000)
        np.random.seed(seed=random_seed)

        for (dataset_bootstraps, timeseries_bootstraps) in zip(dataset_bootstrap_list, timeseries_bootstrap_list):

            dataset_bootstraps = int(dataset_bootstraps)
            if dataset_bootstraps == 0:
                bootstrap_list = [0]
            else:
                bootstrap_list = [dataset_bootstraps] * dataset_bootstraps

            for similarity_metric in similarity_metric_list:
                for cluster_method in cluster_methods:
                    for blocklength in blocklength_list:
                        for n_clusters in n_clusters_list:

                            ism_gsm_stability = []

                            for output_size in output_sizes:
                                
                                experiment_dir = out_dir.format(
                                    analysis=analysis_ID,
                                    rerun=rerun + 1,
                                    output_size=output_size,
                                    similarity_metric=similarity_metric,
                                    cluster_method=cluster_method,
                                    n_clusters=n_clusters,
                                    timeseries_bootstraps=timeseries_bootstraps,
                                    blocklength=blocklength
                                )
                                
                                run_basc_workflow(
                                    subject_file_list, roi_mask_file,
                                    dataset_bootstraps, timeseries_bootstraps,
                                    n_clusters, output_size, bootstrap_list,
                                    proc_mem, similarity_metric,
                                    group_dim_reduce=group_dim_reduce,
                                    cross_cluster=cross_cluster,
                                    cross_cluster_mask_file=cross_cluster_mask_file,
                                    blocklength=blocklength,
                                    affinity_threshold=affinity_threshold,
                                    cluster_method=cluster_method,
                                    out_dir=experiment_dir,
                                    run=run
                                )

                                ism_gsm_stability.append(
                                    np.load(experiment_dir + '/workflow_output/ism_gsm_corr_file/ism_gsm_corr.npy')
                                )

                                ind_clust_stab_mat = np.load(
                                    experiment_dir + '/workflow_output/ind_group_cluster_stability_set/ind_group_cluster_stability_set.npy'
                                )

                                ind_clust_stab_summary = np.concatenate(
                                    (
                                        ind_clust_stab_summary,
                                        np.array([[
                                            n_clusters,
                                            output_size,
                                            ind_clust_stab_mat.mean(),
                                            scipy.stats.variation(ind_clust_stab_mat).mean(),
                                            (ind_clust_stab_mat.mean() - scipy.stats.variation(ind_clust_stab_mat).mean())
                                        ]])
                                    )
                                )

                                # Run Group ClusterMaps
                                gsm_file = experiment_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/group_stability_matrix.npz'
                                clusters_G_file = experiment_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/clusters_G.npy'
                                os.chdir(experiment_dir +'/workflow_output/basc_workflow_runner/basc/join_group_stability/')
                                create_group_cluster_maps(gsm_file, clusters_G_file, roi_mask_file)

                                # Run IGCM on all individual subjects
                                clustvoxscoredir = experiment_dir + '/workflow_output/basc_workflow_runner/basc/individual_group_clustered_maps/mapflow/'
                                clusters_G_file = experiment_dir + '/workflow_output/basc_workflow_runner/basc/join_group_stability/clusters_G.npy'
                            
                            ism_gsm_stability_file = os.path.join(experiment_dir, 'ism_gsm_stability_' + str(n_clusters) + '.npy')
                            np.save(ism_gsm_stability_file, ism_gsm_stability)
                    
                        ind_clust_stab_summary_file = os.path.join(experiment_dir, 'ind_clust_stab_summary.npy')
                        np.save(ind_clust_stab_summary_file, ind_clust_stab_summary)
