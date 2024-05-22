# Online Learning under Haphazard Input Conditions: A Comprehensive Review and Analysis

## Citation

Please consider citing the below paper, if you are using the code provided in this repository.
```
@article{agarwal2024online,
  title={Online Learning under Haphazard Input Conditions: A Comprehensive Review and Analysis},
  author={Agarwal, Rohit and Das, Arijit and Horsch, Alexander and Agarwal, Krishna and Prasad, Dilip K},
  journal={arXiv preprint arXiv:2404.04903},
  year={2024}
}
```

## Overview
This repository contains datasets and implementation codes of different models for the paper, titled "Online Learning under Haphazard Input Conditions: A Comprehensive Review and Analysis".

## File Structure of the Directory

[HaphazardInputsReview/](https://github.com/Rohit102497/HaphazardInputsReview)  
┣ [Code/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code)  
┃ ┣ [AnalyseResults/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/AnalyseResults)  
┃ ┣ [Config/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/Config)   
┃ ┣ [DataCode/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/DataCode)   
┃ ┣ [Models/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/Models)  
┃ ┣ [main.py](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/main.py)  
┃ ┣ [read_results.py](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/read_results.py)  
┃ ┗ [requirements.txt](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Code/requirements.txt)  
┣ [Data/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Data)  
┣ [Results/](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/Results)  
┣ [.gitignore](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/.gitignore)  
┗ [README.md](https://github.com/Rohit102497/HaphazardInputsReview/tree/README/README.md)  


## Datasets
We use 20 different datasets for this project. The link of all the datasets can be found below. Moreover, some of the datasets are also given in their respective folders inside `Data/` directory. To run them, please download the datsets files form the given link below and place them inside their respective directories (see instructions for each dataset below...).  

<p align="center">
Small Datsets
</p>  
<hr>

- ### WPBC
    Data link: https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic  
    Directory: `DataStorage/wbc/`  
    (provided in repository/not provided in repository)  

- ### ionosphere
    Data link: https://archive.ics.uci.edu/dataset/52/ionosphere  
    Directory: `DataStorage/ionosphere/`  
    (provided in repository/not provided in repository)  

- ### WDBC
    Data link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
    Directory: `DataStorage/wdbc/`  
    (provided in repository/not provided in repository)  

- ### australian
    Data link: https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval  
    Directory: `DataStorage/australian/`  
    (provided in repository/not provided in repository)  

- ### WBC
    Data link: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original  
    Directory: `DataStorage/wbc/`  
    (provided in repository/not provided in repository)  

- ### diabetes-f
    Data link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
    Directory: `DataStorage/diabetes_f/`  
    (provided in repository/not provided in repository)  
    **Instructions**: After downloading the file change it's name from `diabetes.csv` to `diabetes_f.csv` 

- ### german
    Data link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data  
    Directory: `DataStorage/german`  
    (provided in repository/not provided in repository)  

- ### IPD
    Data link: https://www.timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand  
    Directory: `DataStorage/ipd`  
    (provided in repository/not provided in repository)  
    Instructions: Download the dataset from the link, and place the files `ItalyPowerDemand_TEST.txt` and `ItalyPowerDemand_TRAIN.txt` inside the directory.   

- ### svmguide3
    Data link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#svmguide3  
    Directory: `DataStorage/svmguide3`  
    (provided in repository/not provided in repository)  

- ### kr-vs-kp
    Data link: https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn  
    Directory: `DataStorage/krvskp`  
    (provided in repository/not provided in repository)  

- ### spambase
    Data link: https://archive.ics.uci.edu/dataset/94/spambase  
    Directory: `DataStorage/spambase`  
    (provided in repository/not provided in repository)  

- ### spamassasin
    Data link: https://spamassassin.apache.org/old/publiccorpus/  
    Directory: `DataStorage/spamassasin`  
    (provided in repository/not provided in repository) 

>Note: Two more small datasets, used for analysis namely, crowdsense(c3) and crowdsense(c5), are not provided due to their unavailability in public domain.

<p align="center">
Medium Datsets
</p> 
<hr>

- ### magic04
    Data link: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope  
    Directory: `DataStorage/magic04`  
    (provided in repository/not provided in repository)  

- ### imdb
    Data link: https://ai.stanford.edu/~amaas/data/sentiment/  
    Directory: `DataStorage/imdb`  
    (provided in repository/not provided in repository)  

- ### a8a
    Data link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a8a  
    Directory: `DataStorage/a8a`  
    (provided in repository/not provided in repository)  

<p align="center">
Large Datsets
</p>   
<hr>

- ### diabets_us
    Data link: Supplementary Material at https://www.hindawi.com/journals/bmri/2014/781670/#supplementary-materials  
    Directory: `DataStorage/diabetes_us`  
    (provided in repository/not provided in repository)  

- ### SUSY
    Data link: https://archive.ics.uci.edu/dataset/279/susy  
    Directory: `DataStorage/susy`  
    (provided in repository/not provided in repository)  

- ### HIGGS
    Data link: https://archive.ics.uci.edu/dataset/280/higgs  
    Directory: `DataStorage/higgs`  
    (provided in repository/not provided in repository) 

## Raw Data Transformation
Some of the data sets need to be cleaned and processesed before they can be used in the models for inference. Details of how to process those datasets are given below.

- Spamassasin
    - Download the files form the link provided and unzip them.  
    - Use the sctipt `Code\DataStorage\DataPreparation\data_spamassasin_conversion.py` to clean the data.
    - Modify the 'path' variable at line 13 to the path of the directory where the unzipped files are located.  
    - The data will automatically be saved in the appropriate directory.

- IMDB  
    - Download the files form the link provided and unzip them.  
    - Use the sctipt `Code\DataStorage\DataPreparation\data_imdb_conversion.py` to clean the data.
    - Modify the 'data_path' variable at line 10 to the path of the directory where the unzipped files are located.  
    - The data will automatically be saved in the appropriate directory.

- Diabetes_us  
    - After downloading the dataset from the provided link, follow the instructions at https://www.hindawi.com/journals/bmri/2014/781670/#supplementary-materials to prepare it for analysis

## Dataset Preparation
### Variable P
For synthetic datasets, we varied the availability of each auxiliary input feature independently by a uniform distribution of probability $p$, i.e., each auxilairy feature is available for $100p\%$. For more information about this, follow paper - Aux-Net (https://link.springer.com/chapter/10.1007/978-3-031-30105-6_46)

## Files
To run the models, see `Code/main.py`. All the comparison models can be run from this.  
After running a model on a certain dataset, run `Code/read_results.py` to display and save the evaluation in csv format.  

## Control Parameters

For **main.py** file, 
1. `seed` : Seed value  
_default_ = 2023

2. `type`: The type of the experiment  
_default_="noassumption", type=str,  
_choices_ = ["noassumption", "basefeatures", "bufferstorage"]

<p align="center">
Data Variables
</p>
<hr>

3. `dataname`: The name of the dataset  
_default_ = "wpbc"  
_choices_ = ["all", "synthetic", "crowdsense_c5", "crowdsense_c3" "spamassasin", "imdb", "diabetes_us", "higgs", "susy", "a8a" "magic04", "spambase", "krvskp", "svmguide3", "ipd", "german" "diabetes_f", "wbc", "australian", "wdbc", "ionosphere", "wpbc"]

4. `syndatatype`: The type to create suitable synthetic dataset  
    _default_ = "variable_p"

5. `probavailable`: The probability of each feature being available to create synthetic data  
    _default_ = 0.5, type = float,

6. `ifbasefeat`: If base features are available  
    _default_ = False

<p align="center">
Method Variables
</p>
<hr>

7. `methodname`: The name of the method (model)  
    _default_ = "nb3"  
    _choices_ = ["nb3", "fae", "olvf", "ocds", "ovfm", "dynfo", "orf3v", "auxnet", "auxdrop"]

8. `initialbuffer`: The storage size of initial buffer trainig  
    _default_ = 0

9. `ifimputation`: If some features needs to be imputed  
    _default_ = False
    
10. `imputationtype`: The type of imputation technique to create base features  
    _default_ = 'forwardfill'  
    _choices_ = ['forwardfill', 'forwardmean', 'zerofill']

11. `nimputefeat`: The number of imputation features  
    _default_ = 2   

12. `ifdummyfeat`: If some dummy features needs to be created  
_default_ = False

13. `dummytype`: The type of technique to create dummy base features  
default = 'standardnormal'

14. `ndummyfeat`: The number of dummy features to create'  
    _default_ = 1    

15. `ifAuxDropNoAssumpArchChange`: If the Aux-Drop architecture needs to be changed to handle no assumption
    _default_ = False

16. `nruns`: The number of times a method should runs (For navie Bayes, it would be 1 because it is a deterministic method)  
_default_ = 5  

For **read_results.py** file,
1. `type`: The type of the experiment  
    _default_ ="noassumption"  
    _choices_ = ["noassumption", "basefeatures" "bufferstorage"]  

2. `dataname`: The name of the dataset  
    _default_ = "wpbc"  
    _choices_ = ["synthetic", "real", "crowdsense_c5", "crowdsense_c3", "spamassasin", "imdb", "diabetes_us", "higgs", "susy", "a8a", "magic04", "spambase", "krvskp", "svmguide3", "ipd", "german", "diabetes_f", "wbc", "australian", "wdbc", "ionosphere", "wpbc"]

3. `probavailable`: The probability of each feature being available to create synthetic data  
    _default_ = 0.5

4. `methodname`:  The name of the method  
    _default_ = "nb3"  
    _choices_ = ["nb3", "fae", "olvf", "ocds", "ovfm", "dynfo", "orf3v", "auxnet", "auxdrop"]

## Dependencies
1. numpy
2. torch
3. pandas
4. random
5. tqdm
6. os
7. pickle
8. tdigest (version == 0.5.2.2)
9. statsmodels (version == 0.14.0)

## Running the code

To run the models, change the control parameters accordingly in the **main.py** file and run
```
python Code/main.py
```
Example: To run  model `nb3` on `wpbc` dataset, with probability of available features 0.75, use the code below
```
python Code/main.py --dataname wpbc --probavailable 0.75 --methodname nb3
```
> Note: For `auxnet` , set either of `--ifimputation True` or `--ifdummyfeat True`, and for `auxdrop` set `--ifAuxDropNoAssumpArchChange True` (as these models were modified from their original implementation to support the absence of (previously required) base-feature)
```
python Code/main.py --dataname ionosphere --probavailable 0.75 --methodname auxnet --ifimputation True
```
or
```
python Code/main.py --dataname ionosphere --probavailable 0.75 --methodname auxnet --ifdummyfeat True
```
and
```
python Code/main.py --dataname synthetic --probavailable 0.75 --methodname auxdrop --ifAuxDropNoAssumpArchChange True
```
<hr>

To read the results and save them in .csv format, run **read_results.py** with appropriate control parameters.
```
python Code/read_results.py
```
Example: To read the results of `nb3` on `wpbc` dataset, with probability of available features 0.75, use the code below
```
python Code/read_results.py --dataname wpbc --probavailable 0.75 --methodname nb3
```
