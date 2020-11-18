# ssnal_elastic

This repository contains an efficient python implementation of a SsNAL method to solve the elastic net problem -- see https://arxiv.org/abs/2006.03970



FILES DESCRIPTION:

    ssnal_elastic_core:
      function to run the SsNAL-EN algorithm for one fixed value of c_lam

    ssnal_elastic_tune:
      function to run the SsNAL-EN algorithm for a grid of c_lam and compute the tuning criteria for each of them.

    auxiliary_functions.py
      contains the auxiliary functions called by ssnal_elastic_core and ssnal_elastic_path, including proximal operator functions and conjugate functions.

    expes/main_core.py:
      main file to run ssnal_elastic_core on synthetic data

    expes/main_path.py:
      main file to run ssnal_elastic_path on synthetic data

    expes/main_datasets.py:
      main file to run ssnal_elastic_core on the real data described in the article and contained in the toy_data folder. The user has to select the data to analyze

    expes/toy_data:
      folder containing the used LIBSVM datasets (housing is loaded directly from a python library)



THE FOLLOWING PYTHON PACKAGES ARE REQUIRED:

	- numpy
	- sklearn
	- scipy
	- tqdm


You can install the package by running `pip install -e .` at the root of the repository, i.e. where the `setup.py` file is.