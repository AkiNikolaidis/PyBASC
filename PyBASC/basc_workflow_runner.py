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
    
    #config.enable_debug_mode()
    config.set('execution', 'keep_inputs', 'true')
    workflow = pe.Workflow(name='basc_workflow_runner')

    if not out_dir:
        out_dir = os.getcwd()

    workflow_dir = os.path.join(out_dir, "workflow_output")
    workflow.base_dir = workflow_dir

    resource_pool = {}

    basc = create_basc(name='basc')
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
    resource_pool['ism_gsm_corr_file'] = (basc, 'outputspec.ism_gsm_corr_file')
    resource_pool['gsclusters_img'] = (basc, 'outputspec.gsclusters_img')
    resource_pool['cluster_voxel_scores_img'] = (basc, 'outputspec.cluster_voxel_scores_img')
    resource_pool['cluster_voxel_scores'] = (basc, 'outputspec.cluster_voxel_scores')
    resource_pool['ind_group_cluster_stability'] = (basc, 'outputspec.ind_group_cluster_stability')
    resource_pool['ind_group_cluster_stability_set'] = (basc, 'outputspec.ind_group_cluster_stability_set')

    ds = pe.Node(nio.DataSink(), name='datasink_workflow_name')
    ds.inputs.base_directory = workflow_dir
    
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

    # TODO @AKI is there any occasion in which running will be false?
    if run == True:
        workflow.run(plugin=plugin, plugin_args=plugin_args)
        outpath = glob.glob(os.path.join(workflow_dir, "*", "*"))

        return outpath
    else:
        return workflow, workflow.base_dir
