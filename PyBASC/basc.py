import os
import sys

import nibabel as nb
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import numpy as np


def group_dim_reduce(
    subjects_files,
    roi_mask_file, compression_dim, group_dim_reduce=False,
    cross_cluster=False, cxc_roi_mask_file=None
):
    if not group_dim_reduce:

        compressor = None
        cxc_compressor = None
        compression_labels_file = None

        return compressor, cxc_compressor, compression_labels_file

    else:
        import numpy as np
        import nibabel as nb
        import pandas as pd
        from sklearn.preprocessing import normalize

        import PyBASC.utils as utils

        roi_mask_img = nb.load(roi_mask_file)
        roi_mask_data = roi_mask_img.get_data().astype('bool')

        x, y, z = nb.load(subjects_files[0]).shape[0:3]

        # TODO @AKI why starting with zeros and appending the data to it?
        group_data = np.zeros((x, y, z, 1))[roi_mask_data]

        for subject_file in subjects_files:
            subject_data = nb.load(subject_file) \
                             .get_data() \
                             .astype('float16')[roi_mask_data]
            group_data = np.append(group_data, subject_data, axis=1)

        # TODO @AKI why selecting 1: ?
        group_data = group_data[:, 1:]
        group_data = normalize(group_data, norm='l2')

        compression = utils.data_compression(group_data.T, roi_mask_img,
                                             roi_mask_data, compression_dim)

        compression_labels = pd.DataFrame(compression['labels'])
        compression_labels = np.array(compression_labels)

        compression_labels_file = './compression_labels.npy'
        np.save(compression_labels_file, compression_labels)
        compression_labels_file = [compression_labels_file]

        compressor = compression['compressor']

        if cross_cluster:

            cxc_roi_mask_image = nb.load(cxc_roi_mask_file)
            cxc_roi_mask_data = cxc_roi_mask_image.get_data().astype('bool')

            cxc_group_data = np.zeros((x, y, z, 1))[cxc_roi_mask_data]

            for subject_file in subjects_files:
                subject_data = nb.load(subject_file) \
                                 .get_data() \
                                 .astype('float16')[cxc_roi_mask_data]

                cxc_group_data = np.append(cxc_group_data, subject_data, axis=1)

            cxc_group_data = cxc_group_data[:, 1:]
            cxc_group_data = normalize(cxc_group_data, norm='l2')

            cxc_compression_dim = compression_dim + 5

            cxc_compression = utils.data_compression(
                cxc_group_data.T,
                cxc_roi_mask_image,
                cxc_roi_mask_data,
                cxc_compression_dim
            )

            cxc_compressor = cxc_compression['compressor']

        else:

            cxc_compressor = None

        return compressor, cxc_compressor, compression_labels_file


def nifti_individual_stability(
    subject_file, roi_mask_file,
    n_bootstraps, n_clusters, compression_dim, similarity_metric,
    compressor=None, cross_cluster=False, cxc_compressor=None,
    cxc_roi_mask_file=None, blocklength=1, cbb_block_size=None,
    affinity_threshold=0.0, cluster_method='ward'
):
    # TODO @AKI update docs
    """
    Calculate the individual stability matrix for a single subject by using Circular Block Bootstrapping method
    for time-series data.

    Parameters
    ----------
    subject_file : string
        Nifti file of a subject
    roi_mask_file : string
        Region of interest (this method is too computationally intensive to perform on a whole-brain volume)
    n_bootstraps : integer
        Number of bootstraps
    n_clusters : integer
        Number of clusters
    cbb_block_size : integer, optional
        Size of the time-series block when performing circular block bootstrap
    affinity_threshold : float, optional
        Minimum threshold for similarity matrix based on correlation to create an edge

    Returns
    -------
    ism : array_like
        Individual stability matrix of shape (`V`, `V`), `V` voxels
    """

    import os
    import numpy as np
    import nibabel as nb
    import PyBASC.utils as utils
    import pandas as pd
    from sklearn.preprocessing import normalize 
    import scipy.sparse

    print('Calculating individual stability matrix of:', subject_file)

    subject_data = nb.load(subject_file).get_data().astype('float32')
    roi_mask_image = nb.load(roi_mask_file)
    roi_mask_data = roi_mask_image.get_data().astype('bool')

    subject_rois = subject_data[roi_mask_data]
    subject_rois = normalize(subject_rois, norm='l2')

    if not compressor:

        compression = utils.data_compression(
            subject_rois.T,
            roi_mask_image,
            roi_mask_data,
            compression_dim
        )

        compressed = compression['compressed']
        compression_labels = pd.DataFrame(compression['labels'])
        compression_labels = np.array(compression_labels)

    else:

        compression_labels = compressor.labels_
        compressed = compressor.transform(subject_rois.T)
        roi_mask_data = None


    compression_labels_file = os.path.join(os.getcwd(), 'compression_labels.npy')
    np.save(compression_labels_file, compression_labels)

    if cross_cluster:

        cxc_roi_mask_img = nb.load(cxc_roi_mask_file)
        cxc_roi_mask_data = cxc_roi_mask_img.get_data().astype('bool')

        subject_cxc_rois = subject_data[cxc_roi_mask_data]
        subject_cxc_rois = normalize(subject_cxc_rois, norm='l2')

        # TODO @AKI why + 5?
        cxc_compression_dim = compression_dim + 5

        if not cxc_compressor:

            cxc_compression = utils.data_compression(
                subject_cxc_rois.T,
                cxc_roi_mask_img,
                cxc_roi_mask_data,
                cxc_compression_dim
            )

            cxc_compressed = cxc_compression['compressed']
            cxc_compression_labels = pd.DataFrame(cxc_compression['labels'])
            cxc_compression_labels = np.array(cxc_compression_labels)

        else:
            cxc_compressor.fit(subject_cxc_rois.T)
            cxc_compression_labels = cxc_compressor.labels_
            cxc_compressed = cxc_compressor.transform(subject_cxc_rois.T)

    else:


        cxc_compressed = None


    ism = utils.individual_stability_matrix(
        compressed, roi_mask_data, n_bootstraps, n_clusters,
        similarity_metric, cxc_compressed, cross_cluster, cbb_block_size,
        blocklength, affinity_threshold, cluster_method
    )

    ism = scipy.sparse.csr_matrix(ism, dtype=np.int8)

    ism_file = os.path.join(os.getcwd(), 'individual_stability_matrix.npz')

    if compressor:
        scipy.sparse.save_npz(ism_file, ism)
    else:
        voxel_ism = utils.expand_ism(ism, compression_labels)
        voxel_ism = voxel_ism.astype("uint8")
        scipy.sparse.save_npz(ism_file, voxel_ism)

    return ism_file, compression_labels_file


def map_group_stability(
    subject_stability_list, n_clusters, bootstrap_list,
    roi_mask_file, group_dim_reduce, cluster_method='ward'
):
    # TODO @AKI review doc
    """
    Calculate the group stability maps for each group-level bootstrap

    Parameters
    ----------
    subject_stability_list : list of strings
        A length `N` list of file paths to numpy matrices of shape (`V`, `V`),
        `N` subjects, `V` voxels
    n_clusters : array_like
        number of clusters extrated from adjacency matrx

    Returns
    -------
    G_file : numpy array
        The group stability matrix for a single bootstrap repitition

    """
    
    import os
    import numpy as np
    import nibabel as nb
    import PyBASC.utils as utils
    import scipy.sparse
    
    print(
        'Calculating group stability matrix for %d subjects' % 
        len(subject_stability_list)
    )

    indiv_stability_set = np.asarray([
        scipy.sparse.load_npz(ism_file).toarray()
        for ism_file in subject_stability_list
    ])


    if bootstrap_list == 1:
        J = indiv_stability_set.mean(axis=0)
    else:
        J = utils.standard_bootstrap(indiv_stability_set).mean(axis=0)

    J = J.astype("uint8")

    if group_dim_reduce:
        roi_mask_img = None
    else:
        roi_mask_img = nb.load(roi_mask_file).get_data().astype('bool')

    G = utils.adjacency_matrix(
        utils.cluster_timeseries(J, roi_mask_img, n_clusters,
                                 similarity_metric='correlation',
                                 affinity_threshold=0.0,
                                 cluster_method=cluster_method)[:, np.newaxis]
    )
    G = G
    G = scipy.sparse.csr_matrix(G, dtype=np.int8)

    G_file = os.path.join(os.getcwd(), 'individual_stability_matrix.npz')
    scipy.sparse.save_npz(G_file, G)

    return G_file


def join_group_stability(
    subject_stability_list, group_stability_list, n_bootstraps, n_clusters,
    roi_mask_file, group_dim_reduce, compression_labels_list,
    cluster_method='ward'
):
    """
    Merges the group stability maps for all and compares to all individual
    stability maps

    Parameters
    ----------
    subject_stability_list : list of strings
        A length `N` list of file paths to numpy matrices of shape (`V`, `V`),
        `N` subjects, `V` voxels
    group_stability_list : list of strings
        A length `N` list of file paths to numpy matrices of shape (`V`, `V`),
        `N` subjects, `V` voxels
    n_bootstraps : array_like
        Number of bootstraps to join and average.
    n_clusters : array_like
        number of clusters extrated from adjacency matrx

    Returns
    -------
    G_file : numpy array
        The group stability matrix for a single bootstrap repitition

    """

    import os
    import numpy as np
    import nibabel as nb
    import PyBASC.utils as utils
    import scipy.sparse


    group_stability_set = np.asarray([
        scipy.sparse.load_npz(G_file) for G_file in group_stability_list
    ])

    gsm = group_stability_set.sum(axis=0)
    G = gsm / n_bootstraps

    G = G * 100
    G = G.astype("uint8")

    if group_dim_reduce:
        compression_labels = np.asarray([np.load(compression_labels_list[0])])
        G = utils.expand_ism(G, compression_labels.T)
        #TODO: Check to see if there's another array type that can be used?
        G = csr_matrix(G, dtype=np.int8).toarray()

    roi_mask_data = nb.load(roi_mask_file).get_data().astype('bool')
    clusters_G = utils.cluster_timeseries(
        G, roi_mask_data, n_clusters,
        similarity_metric='correlation', affinity_threshold=0.0,
        cluster_method=cluster_method
    )
    clusters_G = clusters_G.astype("uint8")

    # TODO @AKI APPLY THIS METHOD TO THE INDIVIDUAL LEVEL CLUSTER
    # TODO @AKI INSERT SECTION HERE TO RETURN ALL OUTPUTS
    #           OF JGSM TO VOXEL RESOLUTION.

    # Cluster labels normally start from 0,
    # so start from 1 to provide contrast when viewing between 0 voxels
    clusters_G += 1

    indiv_stability_set = np.asarray([
        scipy.sparse.load_npz(ism_file) for ism_file in subject_stability_list
    ])

    compression_labels_set = np.asarray([
        np.load(compression_labels_file)
        for compression_labels_file in compression_labels_list
    ])

    ism_gsm_corr = np.zeros(len(subject_stability_list))
    for i in range(len(subject_stability_list)):
        compression_labels = compression_labels_set[i]
        ism = utils.expand_ism(indiv_stability_set[i], compression_labels).toarray()
        ism_gsm_corr[i] = utils.compare_stability_matrices(ism, G)

    gsm_file = os.path.join(os.getcwd(), 'group_stability_matrix.npz')
    scipy.sparse.save_npz(gsm_file, G)

    clusters_G_file = os.path.join(os.getcwd(), 'clusters_G.npy')
    np.save(clusters_G_file, clusters_G)

    ism_gsm_corr_file = os.path.join(os.getcwd(), 'ism_gsm_corr.npy')
    np.save(ism_gsm_corr_file, ism_gsm_corr)

    return G, clusters_G, ism_gsm_corr, \
        gsm_file, clusters_G_file, ism_gsm_corr_file


def ndarray_to_vol(data_array, roi_mask_file, sample_file, filename):
    """
    Converts a numpy array to a nifti file given an roi mask

    Parameters
    ----------
    data_array : array_like
        A data array with the same column length and index alignment as the
        given roi_mask_file.  If data_array is two dimensional, first dimension
        is considered temporal dimension
    roi_mask_file : string
        Path of the roi_mask_file
    sample_file : string or list of strings
        Path of sample nifti file(s) to use for header of the output.
        If list, the first file is chosen.
    filename : string
        Name of output file

    Returns
    -------
    img_file : string
        Path of the nifti file output

    """

    import os
    import numpy as np
    import nibabel as nb

    roi_mask_file = nb.load(roi_mask_file).get_data().astype('bool')

    if data_array.ndim == 1:
        out_vol = np.zeros_like(roi_mask_file, dtype=data_array.dtype)
        out_vol[roi_mask_file] = data_array

    elif data_array.ndim == 2:
        out_vol = np.zeros(
            roi_mask_file.shape[0:3] + [data_array.shape[1]],
            dtype=data_array.dtype
        )
        out_vol[roi_mask_file] = data_array

    else:
        raise ValueError(
            'data_array is %i dimensional, '
            'must be either 1 or 2 dimensional' % len(data_array.shape)
        )

    # TODO @AKI why not use header from ROI file?
    #           it should has the same affine
    if type(sample_file) is list:
        sample_file = sample_file[0]

    nii = nb.load(sample_file)

    img = nb.Nifti1Image(
        out_vol,
        header=nii.get_header(),
        affine=nii.get_affine()
    )

    img_file = os.path.join(os.getcwd(), filename)
    img.to_filename(img_file)

    return img_file, img


def individual_group_clustered_maps(
    subject_stability_list, clusters_G, roi_mask_file,
    group_dim_reduce, compression_labels_file
):
    # TODO @AKI update doc
    """
    Calculate the individual stability maps of each subject based on the group
    stability clustering solution.

    Parameters
    ----------
    subject_stability_list : list of strings
        A length `N` list of file paths to numpy matrices of shape (`V`, `V`),
        `N` subjects, `V` voxels
    clusters_G : array_like
        Length `V` array of cluster assignments for each voxel

    Returns
    -------
    individual_cluster_voxel_scores : list of strings
        A length `N` list of nifti files of the individual group clustered
        stability maps for each cluster.  Temporal dimension of each file
        corresponds to each subject.

    """
    
    import os
    import numpy as np
    import PyBASC.utils as utils
    import PyBASC.basc as basc
    import scipy.sparse
    
    supervox_ism = scipy.sparse.load_npz(subject_stability_list).toarray()

    compression_labels = np.load(compression_labels_file)
    
    if group_dim_reduce==True: 
        indiv_stability_set = utils.expand_ism(supervox_ism, compression_labels).toarray()
    else:
        indiv_stability_set=supervox_ism

    cluster_ids = np.unique(clusters_G)
    cluster_voxel_scores, k_mask = \
        utils.cluster_matrix_average(indiv_stability_set, clusters_G)

    if group_dim_reduce: 
        ind_group_cluster_stability = np.array([
            cluster_voxel_scores[(i-1), clusters_G == i].mean()
            for i in cluster_ids
        ])
    
    cluster_voxel_scores = cluster_voxel_scores.astype("uint8")

    k_mask = k_mask.astype(bool)

    ind_group_cluster_stability_file = os.path.join(
        os.getcwd(), 'ind_group_cluster_stability.npy'
    )
    np.save(ind_group_cluster_stability_file, ind_group_cluster_stability)

    individualized_group_cluster = np.argmax(cluster_voxel_scores, axis=0) + 1
    individualized_group_clusters_file, img = basc.ndarray_to_vol(
        individualized_group_cluster,
        roi_mask_file,
        roi_mask_file,
        os.path.join(os.getcwd(), 'individualized_group_cluster.nii.gz')
    )

    img.to_filename(individualized_group_clusters_file)

    return ind_group_cluster_stability_file, individualized_group_clusters_file


def post_analysis(ind_group_cluster_stability_file_list):
    """
    Creates a composite matrix of all the ind_group_cluster_stability files.
    This allows for choosing the combination of parameters that produces the
    most stable clustering.
    
    Parameters
    ----------
    
    ind_group_cluster_stability_file_list : A list of all
                                            ind_group_cluster_stability files 
     
    Returns
    -------
    ind_group_cluster_stability_set_file : a composite matrix of all the
                                           ind_group_cluster_stability metrics
    """
    
    import os
    import numpy as np

    ind_group_cluster_stability_set = np.asarray([
        np.load(ind_group_cluster)
        for ind_group_cluster in ind_group_cluster_stability_file_list
    ])
    
    ind_group_cluster_stability_set_file = os.path.join(
        os.getcwd(), 'ind_group_cluster_stability_set.npy'
    )
    np.save(
        ind_group_cluster_stability_set_file,
        ind_group_cluster_stability_set
    )
    
    return ind_group_cluster_stability_set_file


# TODO @AKI unused?
def save_igcm_nifti(cluster_voxel_scores_file,clusters_G_file,roi_mask_file):
    """
    Loops through every row of cluster_voxel_scores and creates nifti files
    
    Parameters
    ----------
        cluster_voxel_scores_file- a cluster number by voxel measure of the stability
    of group level solutions at the individual level.
        clusters_G_file- taken from igcm output
        roi_mask_file- file you want to transform the data back into.

    Returns
    -------
    Creates NIFTI files for all the igcm files for each participant across all clusters 
    """
    
    
    import numpy as np
    import PyBASC.basc as basc
    cluster_voxel_scores=np.load(cluster_voxel_scores_file)
    clusters_G=np.load(clusters_G_file)
    cluster_ids = np.unique(clusters_G)
    icvs_idx = 0
    A=[]
    B=[]
    ##import pdb; pdb.set_trace()

    for k in cluster_ids:
        print('k equals \n\n', k, '\n\n') #Loops through every row of cluster_voxel_scores and creates nifti files
        print('clustervoxelscores equals \n\n', cluster_voxel_scores[k-1,:], '\n\n')
        A, B = basc.ndarray_to_vol(cluster_voxel_scores[k-1,:], roi_mask_file, roi_mask_file, 'individual_group_cluster%i_stability.nii.gz' % k)
        print('Output A equals', A, '\n\n')
        #print(B)
#        icvs.append(basc.ndarray_to_vol(cluster_voxel_scores[icvs_idx,:], roi_mask_file, roi_mask_file, 'individual_group_cluster%i_stability.nii.gz' % k))
#        B.to_filename(os.path.join(A))
#        A=[]
#        B=[]


def create_group_cluster_maps(gsm_file, clusters_G_file, roi_mask_file):

    # TODO @AKI update doc
    """
    Loops through every row of cluster_voxel_scores and creates nifti files
    
    Parameters
    ----------
        gsm_file- a cluster number by voxel measure of the stability
            of group level solutions.
        clusters_G_file- taken from igcm output
        roi_mask_file- file you want to transform the data back into.

    Returns
    -------
    Creates NIFTI files for all the gsm file for the group across all clusters 
    
    """
    
    import numpy as np
    import PyBASC.basc as basc
    import PyBASC.utils as utils

    group_stability_set = np.load(gsm_file)
    clusters_G = np.load(clusters_G_file)
    cluster_ids = np.unique(clusters_G)

    group_cluster_voxel_scores, _ = \
        utils.cluster_matrix_average(group_stability_set, clusters_G)

    for k in cluster_ids:
        basc.ndarray_to_vol(
            group_cluster_voxel_scores[k - 1],
            roi_mask_file,
            roi_mask_file,
            'group_level_cluster%i_stability.nii.gz' % k
        )


# TODO @AKI unused?
def ism_nifti(roi_mask_file, n_clusters, out_dir, cluster_method='ward'):
    """
    Calculate the individual level stability and instability maps for
    each of the group clusters. Create Nifti files for each individual
    cluster's stability map
    
    Parameters
    ----------
        roi_mask_file: the mask of the region to calculate stability for.
        n_clusters: the number of clusters calculated
        out_dir: the directory to output the saved nifti images

    Returns
    -------
    Creates NIFTI files for all the ism cluster stability maps
    
    """
    from os import walk
    from os.path import join as pjoin
    import numpy as np
    
    import PyBASC.utils as utils
    
    # TODO FIGURE OUT IF CAN BE ADDED TO BASC WORKFLOW, OR DIFFERENT WORKFLOW?

    ismdir = out_dir + '/workflow_output/basc_workflow_runner/' \
                       'basc/individual_stability_matrices/mapflow/'

    subdirs = next(walk(ismdir))[1]

    roi_mask_data = nb.load(roi_mask_file).get_data().astype('bool')

    for subdir in subdirs:
        subdir = pjoin(ismdir, subdir)

        ism = np.load(pjoin(subdir, 'individual_stability_matrix.npy'))

        clusters_ism = utils.cluster_timeseries(
            ism, roi_mask_data, n_clusters,
            similarity_metric='correlation',
            affinity_threshold=0.0,
            cluster_method=cluster_method
        )

        clusters_ism = clusters_ism + 1

        clusters_nifti_file = pjoin(subdir, 'ism_clust.nii.gz')
        clusters_ism_file = pjoin(subdir, 'clusters_ism.npy')

        # Saving individual level cluster solution
        ndarray_to_vol(clusters_ism, roi_mask_file,
                        roi_mask_file, clusters_nifti_file)

        np.save(clusters_ism_file, clusters_ism)


# TODO @AKI unused?
def gsm_nifti(roi_mask_file, n_clusters, out_dir, cluster_method='ward'):
    """
    Calculate the group level stability and instability maps for each of
    the group clusters.
    Create Nifti files for each individual cluster's stability map
    
    Parameters
    ----------
        roi_mask_file: the mask of the region to calculate stability for.
        n_clusters: the number of clusters calculated
        out_dir: the directory to output the saved nifti images
    """

    import os
    import numpy as np

    import PyBASC.utils as utils
    import PyBASC.basc as basc

    # TODO FIGURE OUT IF CAN BE ADDED TO BASC WORKFLOW, OR DIFFERENT WORKFLOW?
    
    roi_mask_data = nb.load(roi_mask_file).get_data().astype('bool')

    gsmdir = out_dir + \
             '/workflow_output/basc_workflow_runner/basc/join_group_stability/'

    os.chdir(gsmdir)

    gsm = np.load(gsmdir + '/group_stability_matrix.npy')

    clusters_gsm = utils.cluster_timeseries(
        gsm, roi_mask_data, n_clusters,
        similarity_metric='correlation', affinity_threshold=0.0,
        cluster_method=cluster_method
    )

    clusters_gsm = clusters_gsm + 1
    cluster_ids = np.unique(clusters_gsm)

    gsm_cluster_voxel_scores, _ = \
        utils.cluster_matrix_average(gsm, clusters_gsm)
    gsm_cluster_voxel_scores = gsm_cluster_voxel_scores.astype("uint8")

    gsm_cluster_voxel_scores_file = './gsm_cluster_voxel_scores.npy'
    np.save(gsm_cluster_voxel_scores_file, gsm_cluster_voxel_scores)

    # Group cluster stability
    grp_cluster_stability_file = './grp_cluster_stability.npy'
    grp_cluster_stability = np.asarray([
        gsm_cluster_voxel_scores[(k - 1), clusters_gsm == k].mean()
        for k in cluster_ids
    ])
    np.save(grp_cluster_stability_file, grp_cluster_stability)


    # Group cluster instability
    grp_cluster_instability_file = './grp_cluster_instability.npy'
    grp_cluster_instability = np.asarray([
        gsm_cluster_voxel_scores[(k - 1),clusters_gsm!=k].mean()
        for k in cluster_ids
    ])
    np.save(grp_cluster_instability_file, grp_cluster_instability)


    # Group cluster stability difference
    grp_cluster_stability_diff_file = './grp_cluster_stability_diff.npy'
    grp_cluster_stability_diff = grp_cluster_stability - grp_cluster_instability
    np.save(grp_cluster_stability_diff_file, grp_cluster_stability_diff)


    # Group cluster stability volumes
    [
        basc.ndarray_to_vol(
            gsm_cluster_voxel_scores[k - 1],
            roi_mask_file, roi_mask_file,
            'gsm_single_cluster%i_stability.nii.gz' % k
        )
        for k in cluster_ids
    ]
