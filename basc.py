import os
import numpy as np
import nibabel as nb
import sys
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import imp
from os.path import expanduser

def map_group_stability(indiv_stability_list, n_clusters, bootstrap_list, stratification=None):
    import os
    import numpy as np
    import nibabel as nb
    import utils


    #print( 'Calculating group stability matrix for', len(indiv_stability_list), 'subjects.' )


    indiv_stability_set = np.asarray([np.load(ism_file) for ism_file in indiv_stability_list])
    #print( 'Group stability list dimensions:', indiv_stability_set.shape )

    V = indiv_stability_set.shape[2]

    G = np.zeros((V,V))
    J = utils.standard_bootstrap(indiv_stability_set).mean(0)
    
   
    J=J.astype("uint8")
    
    #print( 'calculating adjacency matrix')
    G = utils.adjacency_matrix(utils.cluster_timeseries(J, n_clusters, similarity_metric = 'correlation', affinity_threshold=0.0)[:,np.newaxis])
    #print("finished calculating group stability matrix")
    
    
    G=G.astype("uint8")
    G_file = os.path.join(os.getcwd(), 'group_stability_matrix.npy')
    np.save(G_file, G)
    #print ('Saving group stability matrix %s' % (G_file))
    
    return G_file
       
    

def join_group_stability(indiv_stability_list, group_stability_list, n_bootstraps, n_clusters):
    import os
    import numpy as np
    import nibabel as nb
    import utils
    
    #print("starting join group stability")
    group_stability_set = np.asarray([np.load(G_file) for G_file in group_stability_list])
    gsm=group_stability_set.sum(axis=0)
    G=gsm/int(n_bootstraps)
    
    G=G*100
    G=G.astype("uint8")

    print( 'calculating clusters_G')
    clusters_G = utils.cluster_timeseries(G, n_clusters, similarity_metric = 'correlation', affinity_threshold=0.0)
    #APPLY THIS METHOD TO THE INDIVIDUAL LEVEL CLUSTER
 
    print( 'calculating cluster_voxel scores' )
        
    #print( '1')
    # Cluster labels normally start from 0, start from 1 to provide contrast when viewing between 0 voxels
    clusters_G += 1
    clusters_G=clusters_G.astype("uint8")
    

    
    
    
    indiv_stability_set = np.asarray([np.load(ism_file) for ism_file in indiv_stability_list])
    #print( '2')
    ism_gsm_corr=np.zeros(len(indiv_stability_list))
    
    for i in range(0,len(indiv_stability_list)):
        ism_gsm_corr[i]=utils.compare_stability_matrices(indiv_stability_set[i], G)
    #print( '3')
    print( 'saving files: G')
    gsm_file = os.path.join(os.getcwd(), 'group_stability_matrix.npy')
    np.save(gsm_file, G)
    
    print( 'saving files: clusters_G' )
    clusters_G_file = os.path.join(os.getcwd(), 'clusters_G.npy')
    np.save(clusters_G_file, clusters_G)
        
    print( 'saving files: ism_gsm_corr')
    ism_gsm_corr_file = os.path.join(os.getcwd(), 'ism_gsm_corr.npy')
    np.save(ism_gsm_corr_file, ism_gsm_corr)
    
    return G, clusters_G, ism_gsm_corr, gsm_file, clusters_G_file, ism_gsm_corr_file


def individual_group_clustered_maps(indiv_stability_list, clusters_G, roi_mask_file):
    """
    Calculate the individual stability maps of each subject based on the group stability clustering solution.

    Parameters
    ----------
    indiv_stability_list : list of strings
        A length `N` list of file paths to numpy matrices of shape (`V`, `V`), `N` subjects, `V` voxels
    clusters_G : array_like
        Length `V` array of cluster assignments for each voxel

    Returns
    -------
    individual_cluster_voxel_scores : list of strings
        A length `N` list of nifti files of the individual group clustered stability maps for each cluster.  Temporal
        dimension of each file corresponds to each subject.

    """
    import os
    import numpy as np
    import utils
    import basc

    #indiv_stability_set = np.asarray([np.load(ism_file) for ism_file in indiv_stability_list])
    indiv_stability_mat = np.asarray([np.load(indiv_stability_list)])
    indiv_stability_set = indiv_stability_mat[0]
    #nSubjects = indiv_stability_set.shape[0]
    nVoxels = indiv_stability_set.shape[1]

    cluster_ids = np.unique(clusters_G)
    nClusters = cluster_ids.shape[0]

#    cluster_voxel_scores = np.zeros((nSubjects,nClusters, nVoxels))
#    k_mask=np.zeros((nSubjects,nVoxels, nVoxels))
    cluster_voxel_scores = np.zeros((nClusters, nVoxels))
    k_mask=np.zeros((nVoxels, nVoxels))
    #for i in range(nSubjects):
    cluster_voxel_scores[:,:], k_mask[:,:] = utils.cluster_matrix_average(indiv_stability_set, clusters_G)
    #clust5[0,clusters_g==1].mean()
    
    
    #cluster_voxel_scores[0,clusters_g==1].mean()
    
    ind_group_cluster_stability=[]
    icvs = []
    icvs_idx = 0
    #for i in range(nSubjects):
    for k in cluster_ids:
        icvs.append(basc.ndarray_to_vol(cluster_voxel_scores[icvs_idx,:], roi_mask_file, roi_mask_file, 'individual_group_cluster%i_stability.nii.gz' % k))
        #ind_group_cluster_stability.append(cluster_voxel_scores[(k-1),clusters_G==k].mean())
        icvs_idx += 1
        
    for i in cluster_ids:
        ind_group_cluster_stability.append(cluster_voxel_scores[(i-1),clusters_G==i].mean())
 
    ind_group_cluster_stability=np.array(ind_group_cluster_stability)
    #ind_group_cluster_stability=np.array([1,2,3,4,5])
    #print( 'saving files: icvs')
    icvs_file = os.path.join(os.getcwd(), 'icvs.npy')
    np.save(icvs_file, icvs)
    
    #print( 'saving files: cluster_voxel_scores')
    cluster_voxel_scores=cluster_voxel_scores.astype("uint8")
    cluster_voxel_scores_file = os.path.join(os.getcwd(), 'cluster_voxel_scores.npy')
    np.save(cluster_voxel_scores_file, cluster_voxel_scores)
    
    #print( 'saving files: k_mask')
    k_mask=k_mask.astype("bool_")
    k_mask_file = os.path.join(os.getcwd(), 'k_mask.npy')
    np.save(k_mask_file, k_mask)
    
    #print( 'saving files: ind_group_cluster_stability')
    #ind_group_cluster_stability=ind_group_cluster_stability.astype("uint8")
    ind_group_cluster_stability_file = os.path.join(os.getcwd(), 'ind_group_cluster_stability.npy')
    np.save(ind_group_cluster_stability_file, ind_group_cluster_stability)


    

    return  icvs_file, cluster_voxel_scores_file, k_mask_file, ind_group_cluster_stability_file #icvs, cluster_voxel_scores, k_mask

def post_analysis(ind_group_cluster_stability_file_list):
    import os
    import numpy as np
    ind_group_cluster_stability_set = np.asarray([np.load(ind_group_cluster) for ind_group_cluster in ind_group_cluster_stability_file_list])
    
    
#    ind_group_cluster_stability_mat = np.asarray([np.load(ind_group_cluster_stability_file_list)])
#    ind_group_cluster_stability_set = ind_group_cluster_stability_mat[0]
    
    ind_group_cluster_stability_set_file = os.path.join(os.getcwd(), 'ind_group_cluster_stability_set.npy')
    np.save(ind_group_cluster_stability_set_file, ind_group_cluster_stability_set)
    return ind_group_cluster_stability_set_file
    

def nifti_individual_stability(subject_file, roi_mask_file, n_bootstraps, n_clusters, output_size, cross_cluster=False, roi2_mask_file=None, cbb_block_size=None, affinity_threshold=0.5):
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
    import utils
    import pandas as pd
    import sklearn as sk
    

    #print( 'Calculating individual stability matrix of:', subject_file)


    data = nb.load(subject_file).get_data().astype('float32')
    #print( 'Data Loaded')

    if (roi2_mask_file != None):
       #print( 'Setting up NIS')
        roi_mask_file_nb = nb.load(roi_mask_file)
        roi2_mask_file_nb= nb.load(roi2_mask_file)

        roi_mask_nparray = nb.load(roi_mask_file).get_data().astype('float32').astype('bool')
        roi2_mask_nparray = nb.load(roi2_mask_file).get_data().astype('float32').astype('bool')


        roi1data = data[roi_mask_nparray]
        roi2data = data[roi2_mask_nparray]
        
        #add code that uploads the roi1data and roi2data, divides by the mean and standard deviation of the timeseries
        roi1data=sk.preprocessing.normalize(roi1data, norm='l2')
        roi2data=sk.preprocessing.normalize(roi2data, norm='l2')
        
        #print( 'Compressing data')
        data_dict1 = utils.data_compression(roi1data.T, roi_mask_file_nb, roi_mask_nparray, output_size)
        Y1_compressed = data_dict1['data']
        Y1_compressed = Y1_compressed.T
        Y1_labels = pd.DataFrame(data_dict1['labels'])
        Y1_labels=np.array(Y1_labels)
        #print( 'Y1 compressed')
        
        #print( 'Compressing Y2')

        data_dict2 = utils.data_compression(roi2data.T, roi2_mask_file_nb, roi2_mask_nparray, output_size)
        Y2_compressed = data_dict2['data']
        Y2_compressed=Y2_compressed.T
        Y2_labels = pd.DataFrame(data_dict2['labels'])
        #print( 'Y2 compressed')
        
        #print('Going into ism')
        ism = utils.individual_stability_matrix(Y1_compressed, n_bootstraps, n_clusters, Y2_compressed, cross_cluster, cbb_block_size, affinity_threshold)
        #ism=ism/n_bootstraps # was already done in ism

        
        #print('Expanding ism')
        voxel_num=roi1data.shape[0]
        voxel_ism = utils.expand_ism(ism, Y1_labels)
      
        
        #voxel_ism=voxel_ism*100 # was already done in ism
        voxel_ism=voxel_ism.astype("uint8")

        ism_file = os.path.join(os.getcwd(), 'individual_stability_matrix.npy')
        np.save(ism_file, voxel_ism)
        #print ('Saving individual stability matrix %s for %s' % (ism_file, subject_file))
        return ism_file

    else:

        roi_mask_nparray = nb.load(roi_mask_file).get_data().astype('float32').astype('bool')

        roi1data = data[roi_mask_nparray]
        #print( '(%i voxels, %i timepoints and %i bootstraps' % (roi1data.shape[0], roi1data.shape[1], n_bootstraps))
        
        roi_mask_file_nb=nb.load(roi_mask_file)
        
        data_dict1 = utils.data_compression(roi1data.T, roi_mask_file_nb, roi_mask_nparray, output_size)
        Y1_compressed = data_dict1['data']
        Y1_compressed = Y1_compressed.T
        Y1_labels = pd.DataFrame(data_dict1['labels'])

        ism = utils.individual_stability_matrix(Y1_compressed, n_bootstraps, n_clusters, cbb_block_size, affinity_threshold)
        #ism=ism/n_bootstraps # was already done in ism
        
        #print('expanding ism')
        voxel_num=roi1data.shape[0]
        voxel_ism = utils.expand_ism(ism, Y1_labels)
        
        #voxel_ism=voxel_ism*100 # was already done in ism
        voxel_ism=voxel_ism.astype("uint8")
        
        ism_file = os.path.join(os.getcwd(), 'individual_stability_matrix.npy')
        np.save(ism_file, ism)
        #print( 'Saving individual stability matrix %s for %s' % (ism_file, subject_file))
        return ism_file
    return ism_file


def ndarray_to_vol(data_array, roi_mask_file, sample_file, filename):
    """
    Converts a numpy array to a nifti file given an roi mask

    Parameters
    ----------
    data_array : array_like
        A data array with the same column length and index alignment as the given roi_mask_file.  If data_array is two dimensional,
        first dimension is considered temporal dimension
    roi_mask_file : string
        Path of the roi_mask_file
    sample_file : string or list of strings
        Path of sample nifti file(s) to use for header of the output.  If list, the first file is chosen.
    filename : string
        Name of output file

    Returns
    -------
    img_file : string
        Path of the nifti file output

    """
    import nibabel as nb
    import numpy as np
    import os
    roi_mask_file = nb.load(roi_mask_file).get_data().astype('float32').astype('bool')
    if(len(data_array.shape) == 1):
        out_vol = np.zeros_like(roi_mask_file, dtype=data_array.dtype)
        out_vol[roi_mask_file] = data_array
    elif(len(data_array.shape) == 2):
        out_vol = np.zeros((roi_mask_file.shape[0], roi_mask_file.shape[1], roi_mask_file.shape[2], data_array.shape[0]), dtype=data_array.dtype)
        out_vol[roi_mask_file] = data_array.T
    else:
        raise ValueError('data_array is %i dimensional, must be either 1 or 2 dimensional' % len(data_array.shape) )
    nii = None
    if type(sample_file) is list:
        nii = nb.load(sample_file[0])
    else:
        nii = nb.load(sample_file)
    
    img = nb.Nifti1Image(out_vol, header=nii.get_header(), affine=nii.get_affine())
    img_file = os.path.join(os.getcwd(), filename)
    img.to_filename(img_file)
    
    return img_file


def create_basc(proc_mem, name='basc'):
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
            Minimum threshold for similarity matrix based on correlation to create an edge

    Workflow Outputs::

        outputspec.group_stability_matrix : ndarray
            Group stability matrix
        outputspec.clusters_G: ndarray
            Matrix partitioning each cluster of the group stability matrix
        outputspec.cluster_voxel_scores: ndarray
            Group stability map using gsm and gscluster to calculate average within-cluster stability
        outputspec.gsclusters_img : string (nifti file)
            3-D volume of brain regions partitioned with gsclusters
        outputspec.cluster_voxel_scores_img : string (nifti file)
            3-D volume of brain regions associated with gs_map
        outputspec.individual_cluster_voxel_scores_imgs : list of strings (nifti files)
            3-D volumes of stability scores of each cluster based on group clustering

    BASC Procedure:

    1. Generate individual stability matrices based on multiple clusterings of each bootstrap sample for a single subject
    2. Use stratified bootstrap to sample new datasets of subjects
    3. Calculate average stability matrix of each new dataset using individual stability matrices generated at step 1
    4. Cluster each average stabiilty matrix
    5. Average to create a group stability matrix
    6. Cluster the group stability matrix
    7. Calculate average within-cluster stability based on the clustering of step 6

    Workflow Graph:

    .. image:: ../images/basc.dot.png
        :width: 500

    Detailed Workflow Graph:

    .. image:: ../images/basc_detailed.dot.png
        :width: 500

    References
    ----------
    .. [1] P. Bellec, P. Rosa-Neto, O. C. Lyttelton, H. Benali, and A. C. Evans, "Multi-level bootstrap analysis of stable clusters in resting-state fMRI.," NeuroImage, vol. 51, no. 3, pp. 1126-39, Jul. 2010.

    Examples
    --------
    >>> from CPAC import basc

    """
    import nipype.pipeline.engine as pe


    inputspec = pe.Node(util.IdentityInterface(fields=['subject_file_list',
                                                       'roi_mask_file',
                                                       'dataset_bootstraps',
                                                       'timeseries_bootstraps',
                                                       'n_clusters',
                                                       'output_size',
                                                       'bootstrap_list',
                                                       'proc_mem',
                                                       'cross_cluster',
                                                       'roi2_mask_file',
                                                       'affinity_threshold']),
                        name='inputspec')


    outputspec = pe.Node(util.IdentityInterface(fields=['group_stability_matrix',
                                                        'clusters_G',
                                                        'ism_gsm_corr_file',
                                                        'gsclusters_img',
                                                        'cluster_voxel_scores_img',
                                                        'individual_cluster_voxel_scores_imgs',
                                                        'cluster_voxel_scores',
                                                        'k_mask',
                                                        'ind_group_cluster_stability',
                                                        'ind_group_cluster_stability_set']),
                        name='outputspec')
#G, clusters_G, cluster_voxel_scores, ism_gsm_corr, gsm_file, clusters_G_file, cluster_voxel_scores_file, ism_gsm_corr_file
    basc = pe.Workflow(name=name)



    nis = pe.MapNode(util.Function(input_names=['subject_file',
                                                'roi_mask_file',
                                                'n_bootstraps',
                                                'n_clusters',
                                                'output_size',
                                                'cross_cluster',
                                                'roi2_mask_file',
                                                'cbb_block_size',
                                                'affinity_threshold'],
                                   output_names=['ism_file'],
                                   function=nifti_individual_stability),
                     name='individual_stability_matrices',
                     iterfield=['subject_file',
                                'affinity_threshold'])
    
#    nis.interface.num_threads = proc_mem[0]
    nis.interface.estimated_memory_gb = int(proc_mem[1]/proc_mem[0])


    nis.inputs.cbb_block_size=None

    
    mgsm= pe.MapNode(util.Function(input_names= ['indiv_stability_list',
                                                 'n_clusters',
                                                 'bootstrap_list',
                                                 'stratification'],
                                output_names=['G_file'],
                                function=map_group_stability),
                                name='map_group_stability',
                                iterfield='bootstrap_list')
    
    #mgsm.interface.num_threads = proc_mem[0]
    mgsm.interface.estimated_memory_gb = int(proc_mem[1]/proc_mem[0])

    jgsm= pe.Node(util.Function(input_names=['indiv_stability_list',
                                             'group_stability_list',
                                             'n_bootstraps',
                                             'n_clusters'],
                                output_names=['G',
                                              'clusters_G',
                                              'ism_gsm_corr',
                                              'gsm_file',
                                              'clusters_G_file',
                                              'ism_gsm_corr_file'],
                                function=join_group_stability),
                  name='join_group_stability')
  
    igcm = pe.MapNode(util.Function(input_names=['indiv_stability_list',
                                              'clusters_G',
                                              'roi_mask_file'],
                                 output_names=['icvs_file',
                                               'cluster_voxel_scores_file',
                                               'k_mask_file',
                                               'ind_group_cluster_stability_file'],
                                 function=individual_group_clustered_maps),
                   name='individual_group_clustered_maps',
                   iterfield='indiv_stability_list')
    
    igcm.interface.estimated_memory_gb = int(proc_mem[1]/proc_mem[0])

    post = pe.Node(util.Function(input_names= ['ind_group_cluster_stability_file_list'],
                                          output_names = ['ind_group_cluster_stability_set_file'],
                                          function = post_analysis),
                    name='post_analysis')

    gs_cluster_vol = pe.Node(util.Function(input_names=['data_array',
                                                        'roi_mask_file',
                                                        'sample_file',
                                                        'filename'],
                                           output_names=['img_file'],
                                           function=ndarray_to_vol),
                             name='group_stability_cluster_vol')
#
    gs_score_vol = pe.Node(util.Function(input_names=['data_array',
                                                      'roi_mask_file',
                                                      'sample_file',
                                                      'filename'],
                                         output_names=['img_file'],
                                         function=ndarray_to_vol),
                           name='group_stability_score_vol')



#run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, k_clusters, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)

#node, out_file = resource_pool['anat_reorient']
#workflow.connect([
#    (node, clip_level,      [(out_file, 'in_file')]),
#    (node, mask_skull,      [(out_file, 'in_file_a')]),
#    (node, slice_head_mask, [(out_file, 'infile')])
#])
# 
#workflow.connect([
#    (fill, combine_masks, [('out_file', 'in_file_a')]),
#    (slice_head_mask, combine_masks, [('outfile_path', 'in_file_b')])
#])

###########################################
  

#    basc.connect([
#            (inputspec, nis [('subject_file_list', 'subject_file')]),
#            (inputspec, nis [('roi_mask_file','roi_mask_file')]),
#            (inputspec, nis [('timeseries_bootstraps','n_bootstraps')]),
#            (inputspec, nis [('output_size', 'output_size')]),
#            (inputspec, nis [('cross_cluster','cross_cluster')]),
#            (inputspec, nis [('roi2_mask_file','roi2_mask_file')]),
#            (inputspec, nis [('affinity_threshold','affinity_threshold')]),
#            (inputspec, mgsm [('bootstrap_list','bootstrap_list')]),
#            (inputspec, mgsm [('n_clusters','n_clusters')]),
#            (inputspec, jgsm [('dataset_bootstraps','dataset_bootstraps')]),
#            (inputspec, jgsm [('n_bootstraps','n_bootstraps')]),
#            (inputspec, gs_cluster_vol [('subject_file_list','sample_file')]),
#            (inputspec, gs_cluster_vol [('roi_mask_file','roi_mask_file')]),
#            (inputspec, igcm [('roi_mask_file','roi_mask_file')]),
#            (inputspec, gs_score_vol [('subject_file_list','sample_file')]),
#            (inputspec, gs_score_vol [('roi_mask_file','roi_mask_file')]),
#            (nis, mgsm  [('ism_file','indiv_stability_list')]),
#            (nis, jgsm  [('ism_file','indiv_stability_list')]),
#            (mgsm, jgsm  [('G_file','group_stability_list')]),
#            (nis, igcm  [('ism_file','indiv_stability_list')]),
#            (jgsm, igcm  [('clusters_G','clusters_G')]),
#            (jgsm, gs_cluster_vol  [('clusters_G','data_array')]),
#            ])
#   
#
#    gs_cluster_vol.inputs.filename = 'group_stability_clusters.nii.gz'
#    gs_score_vol.inputs.filename = 'group_stability_scores.nii.gz'

 
    
    

##############################################
    # Gather outside workflow inputs
    basc.connect(inputspec, 'subject_file_list',            nis, 'subject_file')
    basc.connect(inputspec, 'roi_mask_file',                nis, 'roi_mask_file')
    basc.connect(inputspec, 'timeseries_bootstraps',        nis, 'n_bootstraps')
    basc.connect(inputspec, 'n_clusters',                   nis, 'n_clusters')
    basc.connect(inputspec, 'output_size',                  nis, 'output_size')
    basc.connect(inputspec, 'cross_cluster',                nis, 'cross_cluster')
    basc.connect(inputspec, 'roi2_mask_file',               nis, 'roi2_mask_file')
    basc.connect(inputspec, 'affinity_threshold',           nis, 'affinity_threshold')
    basc.connect(inputspec, 'bootstrap_list',               mgsm, 'bootstrap_list')
    basc.connect(inputspec, 'n_clusters',                   mgsm, 'n_clusters')
    basc.connect(inputspec, 'dataset_bootstraps',           jgsm, 'n_bootstraps')
    basc.connect(inputspec, 'n_clusters',                   jgsm, 'n_clusters')
    basc.connect(inputspec, 'subject_file_list',            gs_cluster_vol, 'sample_file')
    basc.connect(inputspec, 'roi_mask_file',                gs_cluster_vol, 'roi_mask_file') 
    basc.connect(inputspec, 'roi_mask_file',                igcm, 'roi_mask_file')
    basc.connect(inputspec, 'subject_file_list',            gs_score_vol, 'sample_file')
    basc.connect(inputspec, 'roi_mask_file',                gs_score_vol, 'roi_mask_file')

    #Node to Node connections
    basc.connect(nis, 'ism_file',                           mgsm, 'indiv_stability_list')
    basc.connect(nis, 'ism_file',                           jgsm, 'indiv_stability_list')
    basc.connect(mgsm, 'G_file',                            jgsm, 'group_stability_list')
    basc.connect(nis, 'ism_file',                           igcm, 'indiv_stability_list')
    basc.connect(jgsm, 'clusters_G',                        igcm, 'clusters_G')
    basc.connect(jgsm, 'clusters_G',                        gs_cluster_vol, 'data_array')
    basc.connect(jgsm, 'clusters_G',                        gs_score_vol, 'data_array')
    basc.connect(igcm, 'ind_group_cluster_stability_file',  post, 'ind_group_cluster_stability_file_list')

    #Outputs
    basc.connect(jgsm, 'gsm_file',                          outputspec, 'group_stability_matrix')
    basc.connect(jgsm, 'clusters_G_file',                   outputspec, 'clusters_G')
    basc.connect(jgsm, 'ism_gsm_corr_file',                 outputspec, 'ism_gsm_corr_file')
    basc.connect(gs_cluster_vol, 'img_file',                outputspec, 'gsclusters_img')
    basc.connect(gs_score_vol, 'img_file',                  outputspec, 'cluster_voxel_scores_img')
    basc.connect(igcm, 'icvs_file',                         outputspec, 'individual_cluster_voxel_scores_imgs')
    basc.connect(igcm, 'cluster_voxel_scores_file',         outputspec, 'cluster_voxel_scores')
    basc.connect(igcm, 'k_mask_file',                       outputspec, 'k_mask')
    basc.connect(igcm, 'ind_group_cluster_stability_file',  outputspec, 'ind_group_cluster_stability')
    basc.connect(post, 'ind_group_cluster_stability_set_file',   outputspec, 'ind_group_cluster_stability_set')



    gs_cluster_vol.inputs.filename = 'group_stability_clusters.nii.gz'
    gs_score_vol.inputs.filename = 'group_stability_scores.nii.gz'

    return basc
