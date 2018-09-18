import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util

from PyBASC.basc import (
    group_dim_reduce,
    nifti_individual_stability,
    map_group_stability,
    join_group_stability,
    individual_group_clustered_maps,
    post_analysis,
    ndarray_to_vol
)


def create_basc(name='basc'):
    """
    Bootstrap Analysis of Stable Clusters (BASC)

    This workflow performs group-level BASC.

    Parameters
    ----------
    name : string, optional
        Name of the workflow.

    Returns
    -------
    basc : nipype.pipeline.engine.Workflow
        BASC workflow.

    Notes
    -----

    Workflow Inputs::

        inputspec.roi : string (nifti file)
            Mask of region(s) of interest
        inputpsec.subjects : list (nifti files)
            4-D timeseries of a group of subjects normalized to MNI space
        inputspec.dataset_bootstraps : integer
            Number of bootstrap samples of the dataset
        inputspec.timeseries_bootstraps : integer
            Number of bootstraps of each subject's timeseries
        inputspec.n_clusters : integer
            Number of clusters at both the individiual and group level
        inputspec.affinity_threshold : list (floats)
            Minimum threshold for similarity matrix based on correlation to
            create an edge

    Workflow Outputs::

        outputspec.group_stability_matrix : ndarray
            Group stability matrix
        outputspec.clusters_G: ndarray
            Matrix partitioning each cluster of the group stability matrix
        outputspec.cluster_voxel_scores: ndarray
            Group stability map using gsm and gscluster to calculate average
            within-cluster stability
        outputspec.gsclusters_img : string (nifti file)
            3-D volume of brain regions partitioned with gsclusters
        outputspec.cluster_voxel_scores_img : string (nifti file)
            3-D volume of brain regions associated with gs_map

    BASC Procedure:

    1. Generate individual stability matrices based on multiple clusterings of
       each bootstrap sample for a single subject
    2. Use stratified bootstrap to sample new datasets of subjects
    3. Calculate average stability matrix of each new dataset using individual
       stability matrices generated at step 1
    4. Cluster each average stabiilty matrix
    5. Average to create a group stability matrix
    6. Cluster the group stability matrix
    7. Calculate average within-cluster stability based on clustering of step 6

    Workflow Graph:

    .. image:: ../images/basc.dot.png
        :width: 500

    Detailed Workflow Graph:

    .. image:: ../images/basc_detailed.dot.png
        :width: 500

    References
    ----------
    .. [1] P. Bellec, P. Rosa-Neto, O. C. Lyttelton, H. Benali, and A. C. Evans,
       "Multi-level bootstrap analysis of stable clusters in resting-state fMRI"
       NeuroImage, vol. 51, no. 3, pp. 1126-39, Jul. 2010.

    Examples
    --------
    >>> from CPAC import basc

    """

    basc = pe.Workflow(name=name)

    inputspec = pe.Node(util.IdentityInterface(fields=[

        'subjects_files',
        'roi_mask_file',

        'dataset_bootstraps',
        'timeseries_bootstraps',
        
        'compression_dim',
        'n_clusters',
        'bootstrap_list',
        'similarity_metric',
        'blocklength',
        'affinity_threshold',
        'cluster_method',
        
        'group_dim_reduce',
        
        'cross_cluster',
        'cxc_roi_mask_file',

    ]), name='inputspec')

    outputspec = pe.Node(util.IdentityInterface(fields=[
        'group_stability_matrix',
        'clusters_G',
        'ism_gsm_corr_file',
        'gsclusters_img',
        'cluster_voxel_scores_img',
        'cluster_voxel_scores',
        'ind_group_cluster_stability',
        'ind_group_cluster_stability_set',
        'individualized_group_clusters_img_file'
    ]), name='outputspec')

    gdr = pe.Node(
        util.Function(
            input_names=['subjects_files',
                         'roi_mask_file',
                         'compression_dim',
                         'group_dim_reduce',
                         'cross_cluster',
                         'cxc_roi_mask_file'],
            output_names=['compressor',
                          'cxc_compressor',
                          'compression_labels_file'],
            function=group_dim_reduce
        ),
        name='group_dim_reduce'
    )

    nis = pe.MapNode(
        util.Function(
            input_names=['subject_file',
                         'roi_mask_file',
                         'n_bootstraps',
                         'n_clusters',
                         'compression_dim',
                         'similarity_metric',
                         'compressor',
                         'cxc_compressor',
                         'cross_cluster',
                         'cxc_roi_mask_file',
                         'cbb_block_size',
                         'blocklength',
                         'affinity_threshold',
                         'cluster_method'],
            output_names=['ism_file', 'compression_labels_file'],
            function=nifti_individual_stability
        ),
        name='individual_stability_matrices',
        iterfield=['subject_file',
                   'affinity_threshold']
    )

    nis.inputs.cbb_block_size = None

    mgsm = pe.MapNode(
        util.Function(
            input_names=['subject_stability_list',
                         'n_clusters',
                         'bootstrap_list',
                         'roi_mask_file',
                         'group_dim_reduce',
                         'cluster_method'],
            output_names=['G_file'],
            function=map_group_stability
        ),
        name='map_group_stability',
        iterfield='bootstrap_list'
    )

    jgsm = pe.Node(
        util.Function(
            input_names=['subject_stability_list',
                         'group_stability_list',
                         'n_bootstraps',
                         'n_clusters',
                         'roi_mask_file',
                         'group_dim_reduce',
                         'compression_labels_list',
                         'cluster_method'],
            output_names=['G',
                          'clusters_G',
                          'ism_gsm_corr',
                          'gsm_file',
                          'clusters_G_file',
                          'ism_gsm_corr_file'],
            function=join_group_stability
        ),
        name='join_group_stability'
    )

    igcm = pe.MapNode(
        util.Function(
            input_names=['subject_stability_list',
                         'clusters_G',
                         'roi_mask_file',
                         'group_dim_reduce',
                         'compression_labels_file'],
            output_names=['ind_group_cluster_stability_file',
                          'individualized_group_clusters_img_file'],
            function=individual_group_clustered_maps
        ),
        name='individual_group_clustered_maps',
        iterfield=['subject_stability_list', 'compression_labels_file']
    )

    post = pe.Node(
        util.Function(
            input_names=['ind_group_cluster_stability_file_list'],
            output_names=['ind_group_cluster_stability_set_file'],
            function=post_analysis
        ),
        name='post_analysis'
    )

    gs_cluster_vol = pe.Node(
        util.Function(
            input_names=['data_array',
                         'roi_mask_file',
                         'sample_file',
                         'filename'],
            output_names=['img_file', 'img'],
            function=ndarray_to_vol
        ),
        name='group_stability_cluster_vol'
    )

    gs_cluster_vol.inputs.filename = 'group_stability_clusters.nii.gz'

    basc.connect([
        (
            inputspec, gdr, [
                ('subjects_files', 'subjects_files'),
                ('roi_mask_file', 'roi_mask_file'),
                ('compression_dim', 'compression_dim'),
                ('cxc_roi_mask_file', 'cxc_roi_mask_file'),
                ('group_dim_reduce', 'group_dim_reduce'),
                ('cross_cluster', 'cross_cluster'),
            ]
        ),
        (
            inputspec, nis, [
                ('subjects_files', 'subject_file'),
                ('roi_mask_file', 'roi_mask_file'),
                ('timeseries_bootstraps', 'n_bootstraps'),
                ('n_clusters', 'n_clusters'),
                ('compression_dim', 'compression_dim'),
                ('similarity_metric', 'similarity_metric'),
                ('cross_cluster', 'cross_cluster'),
                ('cxc_roi_mask_file', 'cxc_roi_mask_file'),
                ('blocklength', 'blocklength'),
                ('affinity_threshold', 'affinity_threshold'),
                ('cluster_method', 'cluster_method'),
            ]
        ),
        (
            inputspec, mgsm, [
                ('bootstrap_list', 'bootstrap_list'),
                ('n_clusters', 'n_clusters'),
                ('roi_mask_file', 'roi_mask_file'),
                ('group_dim_reduce', 'group_dim_reduce'),
                ('cluster_method', 'cluster_method'),
            ]
        ),
        (
            inputspec, jgsm, [
                ('dataset_bootstraps', 'n_bootstraps'),
                ('n_clusters', 'n_clusters'),
                ('roi_mask_file', 'roi_mask_file'),
                ('group_dim_reduce', 'group_dim_reduce'),
                ('cluster_method', 'cluster_method'),
            ]
        ),
        (
            inputspec, gs_cluster_vol, [
                ('subjects_files', 'sample_file'),
                ('roi_mask_file', 'roi_mask_file'),
            ]
        ),
        (
            inputspec, igcm, [
                ('roi_mask_file', 'roi_mask_file'),
                ('group_dim_reduce', 'group_dim_reduce'),
            ]
        ),


        (
            gdr, nis, [
                ('compressor', 'compressor'),
                ('cxc_compressor', 'cxc_compressor'),
            ]
        ),

        (
            nis, mgsm, [
                ('ism_file', 'subject_stability_list'),
            ]
        ),

        (
            nis, jgsm, [
                ('ism_file', 'subject_stability_list'),
                ('compression_labels_file', 'compression_labels_list'),
            ]
        ),

        (
            nis, igcm, [
                ('ism_file', 'subject_stability_list'),
                ('compression_labels_file', 'compression_labels_file'),
            ]
        ),

        (
            mgsm, jgsm, [
                ('G_file', 'group_stability_list'),
            ]
        ),

        (
            jgsm, igcm, [
                ('clusters_G', 'clusters_G'),
            ]
        ),

        (
            jgsm, gs_cluster_vol, [
                ('clusters_G', 'data_array'),
            ]
        ),

        (
            igcm, post, [
                ('ind_group_cluster_stability_file',
                 'ind_group_cluster_stability_file_list'),
            ]
        ),


        # Workflow output
        (
            jgsm, outputspec, [
                ('gsm_file', 'group_stability_matrix'),
                ('clusters_G_file', 'clusters_G'),
                ('ism_gsm_corr_file', 'ism_gsm_corr_file'),
            ]
        ),
        (
            gs_cluster_vol, outputspec, [
                ('img_file', 'gsclusters_img'),
            ]
        ),
        (
            igcm, outputspec, [
                ('ind_group_cluster_stability_file', 'ind_group_cluster_stability'),
                ('individualized_group_clusters_img_file',
                 'individualized_group_clusters_img_file'),
            ]
        ),
        (
            post, outputspec, [
                ('ind_group_cluster_stability_set_file',
                 'ind_group_cluster_stability_set'),
            ]
        ),
    ])

    return basc
