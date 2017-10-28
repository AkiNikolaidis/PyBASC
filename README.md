PyBASC: Bootstrap Analysis of Stable Clusters in Python
============================================================

PyBASC is a free and open source Nipype-based, parcellation pipeline for preprocessed functional MRI data. Designed for use by both novice and expert users, PyBASC allows users to create individual and group level clustering solutions and compare the reliability and reproducibility of these clustering solutions across a wide variety of methods.

## Core Dependencies
[**Python 3.6**](https://www.python.org/download/releases/3.6/)

| Package                                      | Tested version |
|----------------------------------------------|----------------|
| [Scikit-learn](http://scikit-learn.org/)     | 0.18.2         |
| [NumPy](http://www.numpy.org/)               | 1.13.1         |
| [NiBabel](http://nipy.org/nibabel/)          | 2.1.0          |
| [SciPy](http://scipy.org/)                   | 0.19.1         |
| [NiLearn](http://nilearn.github.io/)         | 0.2.6          |
| [NiPype](http://nipype.readthedocs.io/)      | 0.13.1         |


## Installation & Quick Start
------------
- Install from command line using pip
```
pip install basc
```
- Setup PyBASC from command line
```
cd /path/to/PyBASC
python setup.py install
```
## Support
Please use [GitHub issues](https://github.com/AkiNikolaidis/PyBASC/issues) for questions, bug reports or feature requests.


## References
This package is based on the following work:

* Garcia-Garcia, M., Nikolaidis, A., Bellec, P., Craddock, R. C., Cheung, B., Castellanos, F. X., & Milham, M. P. (2017). Detecting stable individual differences in the functional organization of the human basal ganglia. NeuroImage.
* Bellec, P., Rosa-Neto, P., Lyttelton, O. C., Benali, H., & Evans, A. C. (2010). Multi-level bootstrap analysis of stable clusters in resting-state fMRI. Neuroimage, 51(3), 1126-1139.
* Bellec, P., Marrelec, G., & Benali, H. (2008). A bootstrap test to investigate changes in brain connectivity for functional MRI. Statistica Sinica, 1253-1268.

