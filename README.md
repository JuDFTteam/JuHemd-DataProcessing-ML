 Copyright (c) 2022 Peter Grünberg Institut, Forschungszentrum Jülich, Germany

This software publication contains python code which:

1.    Process the JuHemd (see related identifiers) database so that complete data points are selected to be processed in a machine readable and ML-ready format. 
A magnetic threshold of 0.1 Bohr magneton as total absolute magnetic moment is applied to the data. 
Data points with zero Curie temperatures are excluded. 
The script produces data arrays in randomized order and uses supplemental atomic data. 
The supplemental atomic data as well as the generated data set is published separately (see related identifiers). 
The data is generated once including density-functional theory generated descriptors and once excluding them.
This requires the JuHemd.json file to be present when executed.

2.   Is needed to train and evaluate ML-models demonstrating the use of the data for the prediction of Curie temperatures as described in our upcoming paper.
This requires the in step 1 computed data to be acessible by a filepath when executed. 


The software is published at Zenodo using the corresponding Gitlab repository.


[![DOI](https://zenodo.org/badge/579413022.svg)](https://zenodo.org/badge/latestdoi/579413022)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
