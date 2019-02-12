import os
import sys
import numpy as np
import nibabel as nb

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util


def run_basc_workflow(
    subject_file_list, roi_mask_file,
    dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size,
    bootstrap_list, proc_mem, similarity_metric, group_dim_reduce=False,
    cross_cluster=False, cross_cluster_mask_file=None, blocklength=1,
    affinity_threshold=0.0, cluster_method='ward', out_dir=None, run=True
):
    
    """Run the 'template_workflow' function to execute the modular workflow
    with the provided inputs.
    :type input_resource: str
    :param input_resource: The filepath of the { input resource }. Can have
                           multiple.
    :type out_dir: str
    :param out_dir: (default: None) The output directory to write the results
                    to; if left as None, will write to the current directory.
    :type run: bool
    :param run: (default: True) Will run the workflow; if set to False, will
                connect the Nipype workflow and return the workflow object
                instead.
    :rtype: str
    :return: (if run=True) The filepath of the generated anatomical_reorient
             file.
    :rtype: Nipype workflow object
    :return: (if run=False) The connected Nipype workflow object.
    :rtype: str
    :return: (if run=False) The base directory of the workflow if it were to
             be run.
    """

    import os
    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe
    
    from PyBASC.pipeline import create_basc
    from nipype import config
    
    # config.enable_debug_mode()
    config.set('execution', 'keep_inputs', 'true')
    workflow = pe.Workflow(name='basc_workflow_runner')

    if not out_dir:
        out_dir = os.getcwd()

    workflow.base_dir = out_dir

    resource_pool = {}

    basc = create_basc(proc_mem, name='basc')
    basc.inputs.inputspec.set(
        subjects_files=subject_file_list,
        roi_mask_file=roi_mask_file,
        dataset_bootstraps=dataset_bootstraps,
        timeseries_bootstraps=timeseries_bootstraps,
        n_clusters=n_clusters,
        compression_dim=output_size,
        bootstrap_list=bootstrap_list,
        similarity_metric=similarity_metric,
        group_dim_reduce=group_dim_reduce,
        cross_cluster=cross_cluster,
        cxc_roi_mask_file=cross_cluster_mask_file,
        blocklength=blocklength,
        affinity_threshold=affinity_threshold,
        cluster_method=cluster_method
    )
    
    resource_pool['group_stability_matrix'] = (basc, 'outputspec.group_stability_matrix')
    resource_pool['clusters_G'] = (basc, 'outputspec.clusters_G')
    resource_pool['ism_gsm_corr'] = (basc, 'outputspec.ism_gsm_corr')
    resource_pool['gsclusters_img'] = (basc, 'outputspec.gsclusters_img')
    #resource_pool['cluster_voxel_scores_img'] = (basc, 'outputspec.cluster_voxel_scores_img')
    #resource_pool['cluster_voxel_scores'] = (basc, 'outputspec.cluster_voxel_scores')
    resource_pool['ind_group_cluster_stability'] = (basc, 'outputspec.ind_group_cluster_stability')
    resource_pool['individualized_group_clusters'] = (basc, 'outputspec.individualized_group_clusters')
    resource_pool['ind_group_cluster_labels'] = (basc, 'outputspec.ind_group_cluster_labels')
    resource_pool['ind_group_cluster_stability_set'] = (basc, 'outputspec.ind_group_cluster_stability_set')


    ds = pe.Node(nio.DataSink(), name='datasink_workflow_name')
    ds.inputs.base_directory = out_dir
    
    for output in resource_pool.keys():
        node, out_file = resource_pool[output]
        workflow.connect(node, out_file, ds, output)


    plugin = 'MultiProc'
    if int(proc_mem[0]) == 1:
        plugin = 'Linear'

    plugin_args = {
        'n_procs': int(proc_mem[0]),
        'memory_gb': int(proc_mem[1])
    }

    workflow.run(plugin=plugin, plugin_args=plugin_args)
    outpath = glob.glob(os.path.join(out_dir, "*", "*"))
    return outpath


def run_basc_workflow_optimized(
    subject_file_list, roi_mask_file,
    dataset_bootstraps_list, timeseries_bootstraps_list, n_clusters_list, 
    similarity_metric_list, blocklength_list=[1],
    cluster_method_list=['ward'],

    group_dim_reduce=False, output_size_list=[None],

    affinity_threshold_list=[0.0],

    cross_cluster=False, cross_cluster_mask_file=None, 
    out_dir=None, runs=1, proc_mem=None, random_seed=None,
    analysis_id='basc', cache_method='content'
):
    import os
    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe
    from nipype import config
    
    config.enable_debug_mode()
    config.set('execution', 'keep_inputs', 'true')
    if cache_method == 'content':
        config.set('execution', 'hash_method', 'content')
    else:
        config.set('execution', 'hash_method', 'timestamp')
    
    from PyBASC.pipeline import create_basc_optimized
    from PyBASC.utils import generate_random_state

    if not out_dir:
        out_dir = os.getcwd()

    analysis_dir = os.path.join(out_dir, analysis_id)

    rng = np.random.RandomState(random_seed)

    for run_id in range(1, runs + 1):

        rng_run = generate_random_state(rng)

        workflow = pe.Workflow(name='pipeline')
        workflow.base_dir = os.path.join(analysis_dir, 'run_%d' % run_id, 'working')

        basc_workflow = create_basc_optimized(proc_mem, name='basc')

        basc_workflow.inputs.inputspec.set(
            subjects_files=subject_file_list,
            roi_mask_file=roi_mask_file,
            group_dim_reduce=group_dim_reduce,
            cross_cluster=cross_cluster,
            cxc_roi_mask_file=cross_cluster_mask_file,
            random_state_tuple=rng_run.get_state()
        )

        basc_workflow.get_node('inputspec_compression_dim').iterables = [
            ("compression_dim", output_size_list)
        ]
        basc_workflow.get_node('inputspec_boostraps').iterables = [
            ('dataset_bootstraps', dataset_bootstraps_list),
            ('timeseries_bootstraps', timeseries_bootstraps_list),
        ]
        basc_workflow.get_node('inputspec_similarity_metric').iterables = [
            ('similarity_metric', similarity_metric_list)
        ]
        basc_workflow.get_node('inputspec_cluster_method').iterables = [
            ('cluster_method', cluster_method_list)
        ]
        basc_workflow.get_node('inputspec_blocklength').iterables = [
            ('blocklength', blocklength_list)
        ]
        basc_workflow.get_node('inputspec_n_clusters').iterables = [
            ('n_clusters', n_clusters_list)
        ]
        basc_workflow.get_node('inputspec_affinity_threshold').iterables = [
            ('affinity_threshold', affinity_threshold_list)
        ]        

        resource_pool = {}

        resource_pool['group_stability_matrix'] = (basc_workflow, 'outputspec.group_stability_matrix')
        resource_pool['clusters_G'] = (basc_workflow, 'outputspec.clusters_G')
        resource_pool['ism_gsm_corr'] = (basc_workflow, 'outputspec.ism_gsm_corr')
        resource_pool['gsclusters_img'] = (basc_workflow, 'outputspec.gsclusters_img')
        #resource_pool['cluster_voxel_scores_img'] = (basc_workflow, 'outputspec.cluster_voxel_scores_img')
        #resource_pool['cluster_voxel_scores'] = (basc_workflow, 'outputspec.cluster_voxel_scores')
        resource_pool['ind_group_cluster_stability'] = (basc_workflow, 'outputspec.ind_group_cluster_stability')
        resource_pool['individualized_group_clusters'] = (basc_workflow, 'outputspec.individualized_group_clusters')
        resource_pool['ind_group_cluster_labels'] = (basc_workflow, 'outputspec.ind_group_cluster_labels')
        resource_pool['ind_group_cluster_stability_set'] = (basc_workflow, 'outputspec.ind_group_cluster_stability_set')

        ds = pe.Node(nio.DataSink(), name='datasink_workflow_name')
        ds.inputs.base_directory = os.path.join(analysis_dir, 'run_%d' % run_id)
        
        for output in resource_pool.keys():
            node, out_file = resource_pool[output]
            workflow.connect(node, out_file, ds, output)

        
        plugin = 'MultiProc'
        if int(proc_mem[0]) == 1:
            plugin = 'Linear'

        plugin_args = {
            'n_procs': int(proc_mem[0]),
            'memory_gb': int(proc_mem[1])
        }

        # workflow.write_graph(dotfilename='graph.dot', graph2use='exec')
        workflow.run(plugin=plugin, plugin_args=plugin_args)

    return analysis_dir
