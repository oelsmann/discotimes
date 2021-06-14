 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# discotimes
Automated estimation of trends, discontinuities and nonlinearities in vertical land motion using Bayesian Inference. Framework to detecting change points in GNSS, 'altimetry minus tide gauge' and other geophysical time series.


## Rules
Before cloning this repository, please read carefully the following rules:

- The discotimes framework is a work in progress, it is subject to changes
- Please refer to the publication, see [here](#citation)
- Licence: GPLv3


## Prerequisites

### Linux

Install git

    $ sudo apt install git

Install miniconda

    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh

### Windows

Download and install git from [here](https://git-scm.com/downloads).

Download and install Miniconda from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).


## Installation
      pip install git+https://github.com/oelsmann/discotimes 
      conda install -c conda-forge theano-pymc -y
      
## Getting started

Check out the brief [Tutorial](https://github.com/oelsmann/discotimes/blob/master/discotimes_tutorial.md).


## References <span id="citation"><span>
    
    
Oelsmann, J.; Passaro, M.; Sánchez, L.; Dettmering, D.; Schwatke, C.; Seitz, F.; Bayesian modelling of piecewise trends and discontinuities to improve the estimation of coastal vertical land motion. Submitted to Journal of Geodesy, 2021

### Data sources

Blewitt G, Kreemer C, Hammond WC, Gazeaux J (2016) Midas robust trend estimator for accurate gps station velocities without step detection. Journal of Geophysical Research: Solid Earth 121(3):2054–2068, DOI 10.1002/2015JB01255    
    
Caron L, Ivins ER, Larour E, Adhikari S, Nilsson J, Blewitt G (2018) Gia model statistics for grace hydrology, cryosphere, and ocean science. Geophysical Research Letters 45(5):2203– 2212, DOI 10.1002/2017GL076644    
    
Frederikse T, Landerer F, Caron L, Adhikari S, Parkes D, Humphrey V, Dangendorf S, Hogarth P, Zanna L, Cheng L, Wu YH (2020) The causes of sea-level rise since 1900. Nature 584:393–397, DOI 10.1038/s41586-020-2591-3
    
Holgate SJ, Matthews A, Woodworth PL, Rickards LJ, Tamisiea ME, Bradshaw E, Fo-den  PR,  Gordon  KM,  Jevrejeva  S,  Pugh  J  (2013)  New  Data  Systems  and  Products at the  Permanent  Service  for  Mean  Sea  Level.  Journal  of  Coastal  Research  pp  493–504,  DOI  10.2112/JCOASTRES-D-12-00175.1,  URLhttps://doi.org/10.2112/JCOASTRES-D-12-00175.1    
    