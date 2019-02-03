import inspect
import os
import time
from builtins import bytes, str

import nibabel as nb
import nilearn.image as image
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import numpy as np
import scipy as sp
from future import standard_library
from nipype import logging
from nilearn import datasets
from nilearn.image import resample_img, mean_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, show
from nipype.interfaces.base import (BaseInterfaceInputSpec, DynamicTraitedSpec,
                                    Undefined, isdefined, traits)
from nipype.interfaces.io import IOBase, add_traits
from nipype.utils.filemanip import ensure_list
from nipype.utils.functions import create_function_from_source, getsource
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize

standard_library.install_aliases()

iflogger = logging.getLogger('nipype.interface')

max_int = 2 ** 32 - 1


def merge_random_states(*rngs):
    seed = 0.0
    rngs = list(filter(None, rngs))
    for rng in rngs:
        # divide it first to avoid overflow
        seed += generate_random_seed(rng) / len(rngs)
    return np.random.RandomState(int(seed))


def generate_random_seed(rng=None):
    if rng:
        return rng.randint(max_int)
    else:
        return np.random.randint(max_int)


def generate_random_state(rng=None, seed=None):
    if seed:
        seed_rng = np.random.RandomState(seed)
        return merge_random_states(seed_rng, rng)

    seed = generate_random_seed(rng)
    return np.random.RandomState(seed)


def get_random_state(tuple_state=None):
    random_state = np.random.RandomState()
    if tuple_state:
        random_state.set_state(tuple_state)
    return random_state


def timeseries_bootstrap(tseries, block_size, random_state=None):
    """
    Generates a bootstrap sample derived from the input time-series.
    Utilizes Circular-block-bootstrap method described in [1]_.

    Parameters
    ----------
    tseries : array_like
        A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    block_size : integer
        Size of the bootstrapped blocks

    Returns
    -------
    bseries : array_like
        Bootstrap sample of the input timeseries


    References
    ----------
    .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
       changes in brain connectivity for functional MRI. Statistica Sinica,
       special issue on Statistical Challenges and Advances in Brain Science,
       2008, 18: 1253-1268.

    Examples
    --------

    >>> x = np.arange(50).reshape((5, 10)).T
    >>> sample_bootstrap(x, 3)
    array([[ 7, 17, 27, 37, 47 ],
           [ 8, 18, 28, 38, 48 ],
           [ 9, 19, 29, 39, 49 ],
           [ 4, 14, 24, 34, 44 ],
           [ 5, 15, 25, 35, 45 ],
           [ 6, 16, 26, 36, 46 ],
           [ 0, 10, 20, 30, 40 ],
           [ 1, 11, 21, 31, 41 ],
           [ 2, 12, 22, 32, 42 ],
           [ 4, 14, 24, 34, 44 ]])

    """
    import numpy as np

    if not random_state:
        random_state = np.random.RandomState()

    # calculate number of blocks
    k = int(np.ceil(float(tseries.shape[0]) / block_size))

    # generate random indices of blocks
    r_ind = np.floor(random_state.rand(1, k) * tseries.shape[0])
    blocks = np.dot(np.arange(0, block_size)[:, np.newaxis], np.ones([1, k]))

    block_offsets = np.dot(np.ones([block_size, 1]), r_ind)
    block_mask = (blocks + block_offsets).flatten('F')[:tseries.shape[0]]
    block_mask = np.mod(block_mask, tseries.shape[0])

    return tseries[block_mask.astype('int'), :], block_mask.astype('int')


def standard_bootstrap(dataset, random_state=None):
    """
    Generates a bootstrap sample from the input dataset

    Parameters
    ----------
    dataset : array_like
        A matrix of where dimension-0 represents samples

    Returns
    -------
    bdataset : array_like
        A bootstrap sample of the input dataset

    Examples
    --------
    """

    if not random_state:
        random_state = np.random.RandomState()

    n = dataset.shape[0]
    b = random_state.randint(0, high=n - 1, size=n)
    return dataset[b]


def cluster_timeseries(
    X, roi_mask_data, n_clusters, similarity_metric,
    affinity_threshold, cluster_method='ward', random_state=None
):
    """
    Cluster a given timeseries

    Parameters
    ----------
    X : array_like
        A matrix of shape (`N`, `M`) with `N` samples and `M` dimensions
    n_clusters : integer
        Number of clusters
    similarity_metric : {'k_neighbors', 'correlation', 'data'}
        Type of similarity measure for spectral clustering. The pairwise
        similarity measure specifies the edges of the similarity graph.
        'data' option assumes X as the similarity matrix and hence must be
        symmetric.  Default is kneighbors_graph [1]_ (forced to be symmetric)
    affinity_threshold : float
        Threshold of similarity metric when 'correlation' similarity
        metric is used.

    Returns
    -------
    y_pred : array_like
        Predicted cluster labels

    Examples
    --------


    References
    ----------
    .. [1] http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.kneighbors_graph.html

    """
    import numpy as np
    import scipy as sp
    import sklearn as sk
    from sklearn.feature_extraction import image
    from sklearn.cluster import FeatureAgglomeration, SpectralClustering, KMeans

    X = np.array(X)
    X_dist = sp.spatial.distance.pdist(X.T, metric=similarity_metric)
    max_dist = np.nanmax(X_dist)

    X_dist = sp.spatial.distance.squareform(X_dist)
    X_dist[np.isnan(X_dist)] = max_dist

    sim_matrix = 1 - sk.preprocessing.normalize(X_dist, norm='max')
    sim_matrix[sim_matrix < affinity_threshold] = 0

    print("Calculating Hierarchical Clustering") 
     
    if cluster_method == 'ward':

        if roi_mask_data is not None:

            shape = roi_mask_data.shape
            connectivity = image.grid_to_graph(
                n_x=shape[0], n_y=shape[1],
                n_z=shape[2], mask=roi_mask_data
            )

            ward = FeatureAgglomeration(
                n_clusters=n_clusters,
                connectivity=connectivity,
                linkage='ward'
            )
            ward.fit(sim_matrix)
            y_pred = ward.labels_.astype(np.int)

        else:

            ward = FeatureAgglomeration(
                n_clusters=n_clusters,
                affinity='euclidean',
                linkage='ward'
            )
            ward.fit(sim_matrix)
            y_pred = ward.labels_.astype(np.int)

    elif cluster_method == 'spectral':

        spectral = SpectralClustering(
            n_clusters,
            eigen_solver='arpack',
            affinity="precomputed", assign_labels='discretize',
            random_state=random_state
        )
        spectral.fit(sim_matrix)
        y_pred = spectral.labels_.astype(np.int)

    elif cluster_method == 'kmeans':

        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++', n_init=10, random_state=random_state
        )
        kmeans.fit(sim_matrix)
        y_pred = kmeans.labels_.astype(np.int)

    return y_pred


def cross_cluster_timeseries(
    data1, data2, roi_mask_data, n_clusters, similarity_metric,
    affinity_threshold, cluster_method='ward', random_state=None
):
    """
    Cluster a timeseries dataset based on its relationship
    to a second timeseries dataset

    Parameters
    ----------
    data1 : array_like
        A matrix of shape (`N`, `M`) with `N1` samples and `M1` dimensions.
        This is the matrix to receive cluster assignment
    data2 : array_like
        A matrix of shape (`N`, `M`) with `N2` samples and `M2` dimensions.
        This is the matrix with which distances will be calculated to assign
        clusters to data1
    n_clusters : integer
        Number of clusters
    similarity_metric : {'euclidean', 'correlation', 'minkowski', 'cityblock',
                         'seuclidean'}
        Type of similarity measure for distance matrix.  The pairwise similarity
        measure specifies the edges of the similarity graph. 'data' option
        assumes X as the similarity matrix and hence must be symmetric.
        Default is kneighbors_graph [1]_ (forced to be symmetric)
    affinity_threshold : float
        Threshold of similarity metric when 'correlation' similarity metric
        is used.

    Returns
    -------
    y_pred : array_like
        Predicted cluster labels


    Examples
    --------
    np.random.seed(30)
    offset = np.random.randn(30)
    x1 = np.random.randn(200, 30) + 2 * offset
    x2 = np.random.randn(100, 30) + 44 * np.random.randn(30)
    x3 = np.random.randn(400, 30)
    sampledata1 = np.vstack((x1, x2, x3))

    np.random.seed(99)
    offset = np.random.randn(30)
    x1 = np.random.randn(200, 30) + 2 * offset
    x2 = np.random.randn(100, 30) + 44 * np.random.randn(30)
    x3 = np.random.randn(400, 30)
    sampledata2 = np.vstack((x1, x2, x3))

    cross_cluster(sampledata1, sampledata2, 3, 'euclidean')


    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    http://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

    """

    from scipy.spatial.distance import pdist, cdist, squareform
    from sklearn.preprocessing import normalize
    from sklearn.feature_extraction import image
    from sklearn.cluster import FeatureAgglomeration, KMeans, SpectralClustering

    dist_btwn_data_1_2 = np.array(
        cdist(data1.T, data2.T, metric=similarity_metric)
    )

    max_dist = np.nanmax(dist_btwn_data_1_2)
    dist_btwn_data_1_2[np.isnan(dist_btwn_data_1_2)] = max_dist

    dist_of_1 = pdist(
        dist_btwn_data_1_2, metric='euclidean'
    )
    dist_matrix = squareform(dist_of_1)
    sim_matrix = 1 - normalize(dist_matrix, norm='max')
    sim_matrix[sim_matrix < affinity_threshold] = 0

    if cluster_method == 'ward':
        
        if roi_mask_data is not None:
            shape = roi_mask_data.shape
            connectivity = image.grid_to_graph(
                n_x=shape[0], n_y=shape[1],
                n_z=shape[2], mask=roi_mask_data
            )

            ward = FeatureAgglomeration(
                n_clusters=n_clusters,
                connectivity=connectivity,
                linkage='ward'
            )
            ward.fit(sim_matrix)
            y_pred = ward.labels_.astype(np.int)

        else:
            ward = FeatureAgglomeration(
                n_clusters=n_clusters,
                affinity='euclidean',
                linkage='ward'
            )
            ward.fit(sim_matrix)
            y_pred = ward.labels_.astype(np.int)

    elif cluster_method == 'spectral':
        spectral = SpectralClustering(
            n_clusters, eigen_solver='arpack',
            affinity="precomputed", assign_labels='discretize',
            random_state=random_state
        )
        spectral.fit(sim_matrix)
        y_pred = spectral.labels_.astype(np.int)

    elif cluster_method == 'kmeans':
        kmeans = KMeans(
            n_clusters=n_clusters, init='k-means++',
            n_init=10, random_state=random_state
        )

        kmeans.fit(sim_matrix)
        y_pred = kmeans.labels_.astype(np.int)

    return y_pred


def adjacency_matrix(cluster_pred):
    """
    Calculate adjacency matrix for given cluster predictions

    Parameters
    ----------
    cluster_pred : array_like
        A matrix of shape (`N`, `1`) with `N` samples

    Returns
    -------
    A : array_like
        Adjacency matrix of shape (`N`,`N`)

    Examples
    --------
    >>> import numpy as np
    >>> from CPAC.basc import cluster_adjacency_matrix
    >>> x = np.asarray([1, 2, 2, 3, 1])[:, np.newaxis]
    >>> cluster_adjacency_matrix(x).astype('int')
    array([[1, 0, 0, 0, 1],
           [0, 1, 1, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [1, 0, 0, 0, 1]])

    """
    from scipy import sparse
    
    x = cluster_pred.copy()
    if(len(x.shape) == 1):
        x = x[:, np.newaxis]

    # Force the cluster indexing to be positive integers
    if x.min() <= 0:
        x += -x.min() + 1

    A = np.dot(x**-1., x.T) == 1
    A = sparse.csr_matrix(A, dtype=bool)
    return A


def cluster_matrix_average(M, cluster_assignments):
    """
    Calculate the average element value within a similarity matrix for each
    cluster assignment, a measure of within cluster similarity.
    Self similarity (diagonal of similarity matrix) is removed.

    Parameters
    ----------
    M : array_like
    cluster_assignments : array_like

    Returns
    -------
    s : array_like

    Examples
    --------
    >>> import numpy as np
    >>> from CPAC import basc
    >>> S = np.arange(25).reshape(5, 5)
    >>> assign = np.array([0, 0, 0, 1, 1])
    >>> basc.cluster_matrix_average(S, assign)
    array([  6.,   6.,   6.,  21.,  21.])

    """

    # TODO FIGURE OUT TEST FOR THIS FUNCTION

    if np.any(np.isnan(M)):
        raise ValueError('M matrix has NaN values')

    cluster_ids = np.unique(cluster_assignments)
    vox_cluster_label = np.zeros((
        cluster_ids.shape[0],
        cluster_assignments.shape[0]
    ), dtype='float64')

    K_mask = np.zeros(M.shape)

    for s_idx, cluster_id in enumerate(cluster_ids):
        vox_cluster_label[s_idx, :] = \
            M[:, cluster_assignments == cluster_id].mean(axis=1)

        k = (cluster_assignments == cluster_id)[:, np.newaxis].astype(int)

        K = np.dot(k, k.T)
        K[np.diag_indices_from(K)] = 0

        K_mask += K

        # Voxel with its own cluster
        if K.sum() == 0:
            vox_cluster_label[k[:, 0]] = 0.0
        else:
            vox_cluster_label[s_idx, k[:, 0].T] = \
                M[K.astype(bool)].mean()

    return vox_cluster_label, K_mask


def compare_stability_matrices(ism_a, ism_b):
    """
    Calculate the distance between two different stability maps
    
    Parameters
    ----------
    ism_a : array_like
        A numpy stability matrix of shape (`V`, `V`), `V` voxels.
    ism_b : array_like
        A numpy stability matrix of shape (`V`, `V`), `V` voxels.

    Returns
    -------
    similarity : array_like
        The distance between the two input matrices.

    """
    from sklearn.preprocessing import normalize
    from scipy.spatial.distance import correlation

    ism_a = normalize(ism_a, norm='l2')
    ism_b = normalize(ism_b, norm='l2')
    distance = correlation(ism_a.ravel(), ism_b.ravel())
    similarity = 1 - distance

    return similarity


def individual_stability_matrix(
    Y1, roi_mask_data, n_bootstraps, n_clusters, similarity_metric,
    Y2=None, cross_cluster=False, cbb_block_size=None, blocklength=1,
    affinity_threshold=0.0, cluster_method='ward', random_state=None
):
    """
    Calculate the individual stability matrix of a single subject by
    bootstrapping their time-series

    Parameters
    ----------
    Y1 : array_like
        A matrix of shape (`V`, `N`) with `V` voxels `N` timepoints
    Y2 : array_like
        A matrix of shape (`V`, `N`) with `V` voxels `N` timepoints
        For Cross-cluster solutions, this will be the matrix by
        which Y1 is clustered
    n_bootstraps : integer
        Number of bootstrap samples
    k_clusters : integer
        Number of clusters
    cbb_block_size : integer, optional
        Block size to use for the Circular Block Bootstrap algorithm
    affinity_threshold : float, optional
        Minimum threshold for similarity matrix based on correlation
        to create an edge

    Returns
    -------
    S : array_like
        A matrix of shape (`V1`, `V1`), each element v1_{ij} representing
        the stability of the adjacency of voxel i with voxel j
    """

    import numpy as np
    import PyBASC.utils as utils

    if affinity_threshold < 0.0:
        raise ValueError(
            'Affinity_threshold {:d} must be non-negative value'.format(
                affinity_threshold
            )
        )

    N1, V1 = Y1.shape
    temp_block_size = int(np.sqrt(N1))

    if cbb_block_size is None:
        cbb_block_size = int(temp_block_size * blocklength)

    # TODO @AKI review SPATIAL CONSTRAINT EXPERIMENT
    roi_mask_data = None

    S = np.zeros((V1, V1))

    if cross_cluster:
        for _ in range(n_bootstraps):
            if n_bootstraps == 1:
                Y_bootstrap = Y1
                Y_cxc_bootstrap = Y2

            else:
                Y_bootstrap, block_mask = utils.timeseries_bootstrap(
                    Y1, cbb_block_size, random_state=random_state
                )
                Y_cxc_bootstrap = Y2[block_mask.astype('int'), :]

            S += utils.adjacency_matrix(
                utils.cross_cluster_timeseries(
                    Y_bootstrap, Y_cxc_bootstrap, roi_mask_data, n_clusters,
                    similarity_metric=similarity_metric,
                    affinity_threshold=affinity_threshold,
                    cluster_method=cluster_method,
                    random_state=random_state
                )
            )
        
    else:
        for _ in range(n_bootstraps):
            if n_bootstraps == 1:
                Y_bootstrap = Y1

            else:
                Y_bootstrap, _ = utils.timeseries_bootstrap(
                    Y1, cbb_block_size, random_state=random_state
                )

            S += utils.adjacency_matrix(
                utils.cluster_timeseries(
                    Y_bootstrap, roi_mask_data, n_clusters,
                    similarity_metric=similarity_metric,
                    affinity_threshold=affinity_threshold,
                    cluster_method=cluster_method,
                    random_state=random_state
                )[:, np.newaxis]
            )

    S *= 100
    S //= n_bootstraps
    S = S.astype("uint8")

    return S


def expand_ism(ism, Y1_labels):
    """
    Calculates the voxel-wise stability matrix from a
    low dimensional representation.
    
    Parameters
    ----------
    ism : individual stabilty matrix. A symmetric array
    
    Y1_labels : 1-D array of voxel to supervoxel labels, 
                created in initial data compression
    
    Returns
    -------
    A voxel-wise representation of the stabilty matrix.

    """
    import numpy as np
    from scipy import sparse
    voxel_num = len(Y1_labels)
    voxel_ism = np.zeros((voxel_num, voxel_num))

    transform_mat = np.zeros((ism.shape[0], voxel_num))
    for i in range(voxel_num):
        transform_mat[Y1_labels[i], i] = 1
    
    # TODO-  Analyze behavior of sparse matrix ism in the np.dot function
    temp = np.dot(ism.toarray(), transform_mat)
    voxel_ism = np.dot(temp.T, transform_mat)
    sparse_voxel_ism = sparse.csr_matrix(voxel_ism, dtype=np.int8) 

    return sparse_voxel_ism


def data_compression(fmri_masked, mask_img, mask_np, compression_dim):
    # TODO @AKI update doc
    """
    Perform...
    
    Parameters
    ----------
    data : np.ndarray[ndim=2]
           A matrix of shape (`V`, `N`) with `V` voxels `N` timepoints
           The functional dataset that needs to be reduced
    mask : a numpy array of the mask
    compression_dim : integer
        The number of elements that the data should be reduced to

    Returns
    -------
    A dictionaty ...

    """

    from sklearn.feature_extraction import image
    from sklearn.cluster import FeatureAgglomeration

    # Perform Ward clustering
    shape = mask_np.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                       n_z=shape[2], mask=mask_np)

    ward = FeatureAgglomeration(n_clusters=compression_dim,
                                connectivity=connectivity,
                                linkage='ward')

    ward.fit(fmri_masked)

    labels = ward.labels_
    data_reduced = ward.transform(fmri_masked)

    return {
        'compressor': ward,
        'compressed': data_reduced,
        'labels': labels,
    }


class FunctionInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    function_str = traits.Str(mandatory=True, desc='code for function')


class Function(IOBase):
    """Runs arbitrary function as an interface

    Examples
    --------

    >>> func = 'def func(arg1, arg2=5): return arg1 + arg2'
    >>> fi = Function(input_names=['arg1', 'arg2'], output_names=['out'])
    >>> fi.inputs.function_str = func
    >>> res = fi.run(arg1=1)
    >>> res.outputs.out
    6

    """

    input_spec = FunctionInputSpec
    output_spec = DynamicTraitedSpec

    def __init__(self,
                 input_names=None,
                 output_names='out',
                 function=None,
                 imports=None,
                 as_module=False,
                 **inputs):
        """

        Parameters
        ----------

        input_names: single str or list or None
            names corresponding to function inputs
            if ``None``, derive input names from function argument names
        output_names: single str or list
            names corresponding to function outputs (default: 'out').
            if list of length > 1, has to match the number of outputs
        function : callable
            callable python object. must be able to execute in an
            isolated namespace (possibly in concert with the ``imports``
            parameter)
        imports : list of strings
            list of import statements that allow the function to execute
            in an otherwise empty namespace
        """

        super(Function, self).__init__(**inputs)
        if function:
            if as_module:
                module = inspect.getmodule(function).__name__
                full_name = "%s.%s" % (module, function.__name__)
                self.inputs.function_str = full_name
            elif hasattr(function, '__call__'):
                try:
                    self.inputs.function_str = getsource(function)
                except IOError:
                    raise Exception('Interface Function does not accept '
                                    'function objects defined interactively '
                                    'in a python session')
                else:
                    if input_names is None:
                        fninfo = function.__code__
            elif isinstance(function, (str, bytes)):
                self.inputs.function_str = function
                if input_names is None:
                    fninfo = create_function_from_source(function,
                                                         imports).__code__
            else:
                raise Exception('Unknown type of function')
            if input_names is None:
                input_names = fninfo.co_varnames[:fninfo.co_argcount]

        self.as_module = as_module
        self.inputs.on_trait_change(self._set_function_string, 'function_str')
        self._input_names = ensure_list(input_names)
        self._output_names = ensure_list(output_names)
        add_traits(self.inputs, [name for name in self._input_names])
        self.imports = imports
        self._out = {}
        for name in self._output_names:
            self._out[name] = None

    def _set_function_string(self, obj, name, old, new):
        if name == 'function_str':
            if self.as_module:
                module = inspect.getmodule(new).__name__
                full_name = "%s.%s" % (module, new.__name__)
                self.inputs.function_str = full_name
            elif hasattr(new, '__call__'):
                function_source = getsource(new)
                fninfo = new.__code__
            elif isinstance(new, (str, bytes)):
                function_source = new
                fninfo = create_function_from_source(new,
                                                     self.imports).__code__
            self.inputs.trait_set(
                trait_change_notify=False, **{
                    '%s' % name: function_source
                })
            # Update input traits
            input_names = fninfo.co_varnames[:fninfo.co_argcount]
            new_names = set(input_names) - set(self._input_names)
            add_traits(self.inputs, list(new_names))
            self._input_names.extend(new_names)

    def _add_output_traits(self, base):
        undefined_traits = {}
        for key in self._output_names:
            base.add_trait(key, traits.Any)
            undefined_traits[key] = Undefined
        base.trait_set(trait_change_notify=False, **undefined_traits)
        return base

    def _run_interface(self, runtime):
        # Create function handle
        if self.as_module: 
            import importlib 
            pieces = self.inputs.function_str.split('.') 
            module = '.'.join(pieces[:-1]) 
            function = pieces[-1] 
            try: 
                function_handle = getattr(importlib.import_module(module), function) 
            except ImportError: 
                raise RuntimeError('Could not import module: %s' % self.inputs.function_str) 
        else: 
            function_handle = create_function_from_source(self.inputs.function_str, 
                                                          self.imports) 

        # Get function args
        args = {}
        for name in self._input_names:
            value = getattr(self.inputs, name)
            if isdefined(value):
                args[name] = value

        out = function_handle(**args)
        if len(self._output_names) == 1:
            self._out[self._output_names[0]] = out
        else:
            if isinstance(out, tuple) and \
                    (len(out) != len(self._output_names)):
                raise RuntimeError('Mismatch in number of expected outputs')

            else:
                for idx, name in enumerate(self._output_names):
                    self._out[name] = out[idx]

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        for key in self._output_names:
            outputs[key] = self._out[key]
        return outputs
