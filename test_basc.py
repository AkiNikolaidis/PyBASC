import os
import time
from os.path import expanduser
import matplotlib
import numpy as np
import nibabel as nb
import pandas as pd
import nilearn.image as image
import scipy as sp


from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, show
from nilearn.image.image import mean_img
from nilearn.image import resample_img
from matplotlib import pyplot as plt


from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
#

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util


matplotlib.style.use('ggplot')
home = expanduser("~")
proc_mem= [2,4]

#%% TEST UTILS.PY
#Remaining Tests:
    #compare_stability_matrix
    #expand_ism
    #data_compression

def test_timeseries_bootstrap():
    """
    Tests the timeseries_bootstrap method of BASC workflow
    """
    np.random.seed(27)
    #np.set_printoptions(threshold=np.nan)
    
    # Create a 10x5 matrix which counts up by column-wise
    x = np.arange(50).reshape((5,10)).T
    actual = timeseries_bootstrap(x,3)
    desired = np.array([[ 4, 14, 24, 34, 44],
                       [ 5, 15, 25, 35, 45],
                       [ 6, 16, 26, 36, 46],
                       [ 8, 18, 28, 38, 48],
                       [ 9, 19, 29, 39, 49],
                       [ 0, 10, 20, 30, 40],
                       [ 7, 17, 27, 37, 47],
                       [ 8, 18, 28, 38, 48],
                       [ 9, 19, 29, 39, 49],
                       [ 8, 18, 28, 38, 48]])
    np.testing.assert_equal(actual, desired)


def test_standard_bootstrap():
    """
    Tests the standard_bootstrap method of BASC workflow
    """
    np.random.seed(27)
    x = np.arange(50).reshape((5,10)).T
    actual = standard_bootstrap(x)
    desired = np.array([[ 3, 13, 23, 33, 43],
                        [ 8, 18, 28, 38, 48],
                        [ 8, 18, 28, 38, 48],
                        [ 8, 18, 28, 38, 48],
                        [ 0, 10, 20, 30, 40],
                        [ 5, 15, 25, 35, 45],
                        [ 8, 18, 28, 38, 48],
                        [ 1, 11, 21, 31, 41],
                        [ 2, 12, 22, 32, 42],
                        [ 1, 11, 21, 31, 41]])
    np.testing.assert_equal(actual, desired)

def test_adjacency_matrix():
    """
    Tests the adjacency_matrix of BASC workflow
    """
    x = np.asarray([1, 2, 2, 3, 1])[:,np.newaxis]
    actual = adjacency_matrix(x).astype(int)
    desired = np.array([[1, 0, 0, 0, 1],
                       [0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 1]])
    np.testing.assert_equal(actual, desired)

def generate_blobs():
    np.random.seed(27)
    offset = np.random.randn(30)

    x1 = np.random.randn(200,30) + 2*offset
    x2 = np.random.randn(100,30) + 44*np.random.randn(30)
    x3 = np.random.randn(400,30)
    blobs = np.vstack((x1,x2,x3))
    return blobs


def generate_simple_blobs(x):
    np.random.seed(x)
    offset = np.random.randn(30)

    x1 = np.random.randn(200,30) + 2*offset
    x2 = np.random.randn(100,30) + 44*np.random.randn(30)+ 2*offset

    blobs = np.vstack((x1,x2))
    return blobs



def generate_blobs_3d():
    np.random.seed(27)
    x1 = np.random.randn(200,3) + np.array([1.4, 1.8, 22.2])
    x2 = np.random.randn(100,3) + np.array([4.7, 4.0, 9.6])
    x3 = np.random.randn(400,3) + np.array([100.7, 100.0, 100.8])
    blobs = np.vstack((x1,x2,x3))
    return blobs

def test_cluster_timeseries():
    """
    Tests the cluster_timeseries method on three blobs in three dimensions (to make correlation possible)
    """
    blobs = generate_blobs_3d()
    y_predict = cluster_timeseries(blobs, 3, similarity_metric = 'correlation')


def test_cross_cluster_timeseries():
    np.random.seed(30)
    offset = np.random.randn(30)
    x1 = np.random.randn(20,30) + 10*offset
    x2 = np.random.randn(10,30) + 44*np.random.randn(30)
    sampledata1 = np.vstack((x1,x2))
    sampledata2 = sampledata1
    actual = cross_cluster_timeseries(sampledata1, sampledata2, 2, 'correlation')
    desired = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_equal(actual,desired)
    print('Correlation equals ', 1-sp.spatial.distance.correlation(actual,desired))


def test_individual_stability_matrix():
    """
    Tests individual_stability_matrix method on three gaussian blobs.
    """

    blobs = generate_blobs()
    ism = individual_stability_matrix(blobs, 10, 3)

    assert False

def test_cross_cluster_individual_stability_matrix():
    """
    Tests individual_stability_matrix method on three gaussian blobs.
    """

    blobs1 = generate_simple_blobs(27)
    blobs2 = generate_simple_blobs(27)
    blobs2 = blobs2[0:150,:]
    ism = individual_stability_matrix(blobs1, 10, 2, Y2 = blobs2, cross_cluster = True)

    return ism

def test_nifti_individual_stability():

    subject_file = home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz'

    roi_mask_file=home + '/C-PAC/CPAC/basc/sampledata/masks/BG.nii.gz'
    roi2_mask_file=home + '/C-PAC/CPAC/basc/sampledata/masks/yeo_2.nii.gz'
    
   
    n_bootstraps=100
    n_clusters=2
    output_size=20
    cross_cluster=True
    
    cbb_block_size=None
    affinity_threshold=0.5
    nifti_individual_stability(subject_file, roi_mask_file, n_bootstraps, n_clusters, output_size, cross_cluster, roi2_mask_file, cbb_block_size, affinity_threshold)


def test_cluster_matrix_average():
    
    import utils
    import basc
    import matplotlib.pyplot as plt
    
    
    blobs = generate_blobs()
    ism = individual_stability_matrix(blobs, 100, 3)
    y_predict = cluster_timeseries(blobs, 3, similarity_metric = 'euclidean')
    cluster_voxel_scores, K_mask = cluster_matrix_average(ism, y_predict)
    
    plt.imshow(K_mask)
    

#%% TEST BASC.PY
#Remaining Tests to write:
    #Join_group_stability
    #cluster_selection
    #individual_group_clustered_maps
    #ndarray_to_vol
    
def new_test_group_stability_matrix():
    """
    Tests group_stability_matrix method.  This creates a dataset of blobs varying only by additive zero-mean gaussian
    noise and calculates the group stability matrix.
    """
    
    import utils
    import basc
    
    bootstrap=20
    blobs = generate_blobs()

    ism_dataset = np.zeros((5, blobs.shape[0], blobs.shape[0]))
    
    indiv_stability_list=[]
    
    for i in range(ism_dataset.shape[0]):
        ism_dataset[i] = utils.individual_stability_matrix(blobs + 0.2*np.random.randn(blobs.shape[0], blobs.shape[1]), 10, 3, affinity_threshold = 0.0)
        f = 'ism_dataset_%i.npy' % i
        indiv_stability_list.append(f)
        np.save(f, ism_dataset[i])

    #indiv_stability_list=ism_list
    n_bootstraps=10
    n_clusters=3

    
    G = basc.map_group_stability(indiv_stability_list, 10, 3)
    
    return G
    
   # for boot in 
#    cluster_G, cluster_voxel_scores, gsm_file, clusters_G_file, cluster_voxel_scores_file
#    
#    def join_group_stability(group_stability_list, n_bootstraps, n_clusters):
#    
#    return G, cluster_G, cluster_voxel_scores, gsm_file, clusters_G_file, cluster_voxel_scores_file


def test_group_stability_matrix():
    """
    Tests group_stability_matrix method.  This creates a dataset of blobs varying only by additive zero-mean gaussian
    noise and calculates the group stability matrix.
    """
    #def map_group_stability(indiv_stability_list, n_clusters, bootstrap_list, stratification=None):

    
    blobs = generate_blobs()
    
    ism_dataset = np.zeros((5, blobs.shape[0], blobs.shape[0]))
    ism_list = []
    for i in range(ism_dataset.shape[0]):
        ism_dataset[i] = individual_stability_matrix(blobs + 0.2*np.random.randn(blobs.shape[0], blobs.shape[1]), 10, 3, affinity_threshold = 0.0)
        f = 'ism_dataset_%i.npy' % i
        ism_list.append(f)
        np.save(f, ism_dataset[i])

    G, cluster_G, cluster_voxel_scores = group_stability_matrix(ism_list, 10, 3, [0,1,1,1,0])

    return G, cluster_g, cluster_voxel_scores


def test_individual_group_clustered_maps():
#    indiv_stability_list
#    clusters_G 
#    roi_mask_file
#
#    import utils
#    import basc
#    
#    bootstrap=20
#    blobs = generate_blobs()
#
#    ism_dataset = np.zeros((5, blobs.shape[0], blobs.shape[0]))
#    
#    indiv_stability_list=[]
#    
#    for i in range(ism_dataset.shape[0]):
#        ism_dataset[i] = utils.individual_stability_matrix(blobs + 0.2*np.random.randn(blobs.shape[0], blobs.shape[1]), 10, 3, affinity_threshold = 0.0)
#        f = 'ism_dataset_%i.npy' % i
#        indiv_stability_list.append(f)
#        np.save(f, ism_dataset[i])
#
#    G, cluster_G, cluster_voxel_scores = group_stability_matrix(ism_list, 10, 3, [0,1,1,1,0])
    
    
    
    import basc
    import utils
    subject_file_list= [home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub3/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz']

    roi_mask_file= home + '/git_repo/BASC/masks/LC_Quarter_Res.nii.gz'
    dataset_bootstraps=50
    timeseries_bootstraps=10
    n_clusters=3
    output_size=10
    bootstrap_list=list(range(0,dataset_bootstraps))
    cross_cluster=True
    roi2_mask_file= home + '/git_repo/BASC/masks/RC_Quarter_Res.nii.gz'
    cbb_block_size=None
    affinity_threshold= 0.5 #* len(subject_file_list)
    out_dir= home + '/BASC_outputs/ClusterCorrTest3'
    run=True
    ismfile=[]
    for i in range(0,len(subject_file_list)):
        temp = basc.nifti_individual_stability(subject_file_list[i], roi_mask_file, timeseries_bootstraps, n_clusters, output_size, cross_cluster, roi2_mask_file, cbb_block_size, affinity_threshold)
        ismfile.append(temp)
    
    G_file=[]
    for i in range(0,dataset_bootstraps):
        temp2= map_group_stability(ismfile, n_clusters, bootstrap_list, stratification=None)
        G_file.append(temp2)
        
    G, clusters_G, cluster_voxel_scores, ism_gsm_corr, gsm_file, clusters_G_file, cluster_voxel_scores_file, ism_gsm_corr_file= basc.join_group_stability(ismfile, G_file, dataset_bootstraps, n_clusters)


    icvs=basc.individual_group_clustered_maps(ismfile, clusters_G, roi_mask_file)


######################NEW FUNCTIONS TO TEST##########################################
######################NEW FUNCTIONS TO TEST##########################################
######################NEW FUNCTIONS TO TEST##########################################
#def test_join_group_stability():
#    #TODO testing the group input from the group stability matrix.
#    #def join_group_stability(group_stability_list, n_bootstraps, n_clusters):
#
#def test_ndarray_to_vol():
#    #TODO testing the transformation from array to volume
#####################NEW FUNCTIONS TO TEST###########################################
#######################NEW FUNCTIONS TO TEST#########################################
########################NEW FUNCTIONS TO TEST########################################






#%% TEST BASC WORKFLOW



def test_basc_workflow_runner():

    from basc_workflow_runner import run_basc_workflow
    import utils
    subject_file_list= [home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub3/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub1/Func_Quarter_Res.nii.gz',
                        home + '/git_repo/BASC/sample_data/sub2/Func_Quarter_Res.nii.gz']

    roi_mask_file= home + '/git_repo/BASC/masks/LC_Quarter_Res.nii.gz'
    dataset_bootstraps=50
    timeseries_bootstraps=10
    n_clusters=3
    output_size=10
    bootstrap_list=list(range(0,dataset_bootstraps))
    cross_cluster=True
    roi2_mask_file= home + '/git_repo/BASC/masks/RC_Quarter_Res.nii.gz'
    affinity_threshold= [0.5] * len(subject_file_list)
    out_dir= home + '/BASC_outputs/ClusterCorrTest5'
    run=True
    
    

    basc_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)









def test_compare_stability_matrices():
    
    import utils
    import basc
    
    
    bootstrap=20
    blobs = generate_blobs()
    n_bootstraps=10
    n_clusters=5
    subjects=20
    
    
    ism_dataset = np.zeros((subjects, blobs.shape[0], blobs.shape[0]))
    ism_list = []
    for i in range(ism_dataset.shape[0]):
        ism_dataset[i] = utils.individual_stability_matrix(blobs + 0.2*np.random.randn(blobs.shape[0], blobs.shape[1]), n_bootstraps, n_clusters, affinity_threshold = 0.0)
        f = 'ism_dataset_%i.npy' % i
        ism_list.append(f)
        np.save(f, ism_dataset[i])

    indiv_stability_list=ism_list
   
    
    G = basc.map_group_stability(ism_list, n_bootstraps, n_clusters)
    
    gsm=np.load(G)
    gsm=gsm.astype("float64")

    corr= []
    corr= np.zeros((subjects,1))
    for j in range(ism_dataset.shape[0]):
        ism=ism_dataset[j].astype("float64")
        corr[j] = utils.compare_stability_matrices(gsm,ism)

    meandist5 = corr.mean()
    vardist5 = corr.var()
    sumdist5 = corr.cumsum()

#%%

def test_basc_workflow_runner():

    from basc_workflow_runner import run_basc_workflow
    import utils
    subject_file_list=    ['/Users/aki.nikolaidis/BGDev_SampleData/A00060280/reduced100.nii.gz',
                           '/Users/aki.nikolaidis/BGDev_SampleData/A00060280/reduced100.nii.gz',
                           '/Users/aki.nikolaidis/BGDev_SampleData/A00060280/reduced100.nii.gz',
                           '/Users/aki.nikolaidis/BGDev_SampleData/A00060384/reduced100.nii.gz',
                           '/Users/aki.nikolaidis/BGDev_SampleData/A00060384/reduced100.nii.gz',
                           '/Users/aki.nikolaidis/BGDev_SampleData/A00060384/reduced100.nii.gz']

    roi_mask_file=home + '/git_repo/basc/masks/BG.nii.gz'
    dataset_bootstraps=10
    timeseries_bootstraps=50
    n_clusters=4
    output_size=500
    bootstrap_list=list(range(0,dataset_bootstraps))
    cross_cluster=True
    roi2_mask_file=home + '/git_repo/basc/masks/yeo_2.nii.gz'
    affinity_threshold= [0.5] * len(subject_file_list)
    out_dir= home + '/BASC_outputs/clusterCorrTest'
    run=True
    
    

    basc_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)






def heavy_basc_workflow_test():

    from basc_workflow_runner import run_basc_workflow
    import utils

    subject_file_list=['/Users/aki.nikolaidis/BGDev_SampleData/A00060846/bandpassed_demeaned_filtered_antswarp.nii.gz',
                         '/Users/aki.nikolaidis/BGDev_SampleData/A00060603/bandpassed_demeaned_filtered_antswarp.nii.gz',
                         '/Users/aki.nikolaidis/BGDev_SampleData/A00060503/bandpassed_demeaned_filtered_antswarp.nii.gz',
                         '/Users/aki.nikolaidis/BGDev_SampleData/A00060429/bandpassed_demeaned_filtered_antswarp.nii.gz',
                         '/Users/aki.nikolaidis/BGDev_SampleData/A00060384/bandpassed_demeaned_filtered_antswarp.nii.gz',
                         '/Users/aki.nikolaidis/BGDev_SampleData/A00060280/bandpassed_demeaned_filtered_antswarp.nii.gz']


    roi_mask_file=home + '/git_repo/basc/masks/BG.nii.gz'
   
    dataset_bootstraps=50
    timeseries_bootstraps=10
    n_clusters=2
    output_size=10
    cross_cluster=True
    
    roi2_mask_file=home + '/git_repo/basc/masks/yeo_2.nii.gz'
    
    affinity_threshold= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    out_dir= home + '/BASC_outputs'
    run=True
    

    basc_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)

def NED_heavy_basc_workflow_test():
#%%

    
    from basc_workflow_runner import run_basc_workflow
    import utils
    import time
    matrixtime = time.time()

#    subject_file_list= ['/data/rockland_sample/A00060603/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00060503/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00060429/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00060384/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00060280/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00059935/functional_mni/_scan_dsc_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00059875/functional_mni/_scan_dsc_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00059734/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz',
#                         '/data/rockland_sample/A00059733/functional_mni/_scan_clg_2_rest_645/bandpassed_demeaned_filtered_antswarp.nii.gz']

    subject_file_list= ['/data/Projects/anikolai/rockland_downsampled/A00018030/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00027159/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00027167/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00027439/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00027443/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00030980/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00030981/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00031216/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00031219/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00031410/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00031411/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00031578/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00031881/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00032008/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00032817/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00033231/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00033714/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00034073/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00034074/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00034350/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035291/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035292/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035364/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035377/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035869/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035940/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035941/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00035945/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00037125/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00037368/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00037458/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00037459/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00037483/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00038603/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00038706/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00039075/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00039159/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00039866/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00040342/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00040440/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00040556/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00040798/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00040800/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00040815/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00041503/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00043240/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00043282/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00043494/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00043740/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00043758/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00043788/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00044084/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00044171/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00050743/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00050847/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00051063/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00051603/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00051690/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00051691/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00051758/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052069/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052165/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052183/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052237/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052461/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052613/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00052614/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00053203/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00053320/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00053390/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00053490/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00053744/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00053873/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00054206/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00055693/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00056703/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00057405/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00057480/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00057725/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00057862/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00057863/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00057967/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058004/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058053/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058060/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058061/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058215/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058229/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058516/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058537/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058570/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058685/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00058951/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059109/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059325/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059427/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059733/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059734/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059865/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059875/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00059935/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00060280/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00060384/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00060429/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00060503/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00060603/3mm_resampled.nii.gz',
    '/data/Projects/anikolai/rockland_downsampled/A00060846/3mm_resampled.nii.gz']

    roi_mask_file=home + '/git_repo/BASC/masks/cerebellum_3mm.nii.gz'
   
    dataset_bootstraps=50
    timeseries_bootstraps=50
    n_clusters=3
    output_size=400
    cross_cluster=True
    bootstrap_list=list(range(0,dataset_bootstraps))
<<<<<<< HEAD
    proc_mem=[10,80]
    
    roi2_mask_file=home + '/git_repo/BASC/masks/yeo_3mm.nii.gz'
=======

    roi2_mask_file=home + '/git_repo/BASC/masks/yeo2_3mm.nii.gz'
>>>>>>> 7e926f8f360768749457ef7b47d0186fef27127c
    
    affinity_threshold= [0.5] * 107 #[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    out_dir=   '/data/Projects/anikolai/BASC_outputs/CouilleCerebellum'
    run=True
    

    basc_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, cross_cluster=cross_cluster, roi2_mask_file=roi2_mask_file, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)

    print((time.time() - matrixtime))
#%%
