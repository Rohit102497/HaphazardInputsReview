# HaphazardInputsReview

## Overview
This repository contains datasets and implementation codes of different models for the paper, titled "Adaptive Online Learning under Haphazard Input
Conditions: A Comprehensive Review, Metrics, and
Evaluation of Methodologies".

## Datasets
We use 20 different datasets for this project. The link of all the datasets can be found below. Moreover, the datasets are also given in their respective folders inside `DataStorage/` directory.
Out of the 20 datasets, X are big datasets, hence they are not provided inside the directory. To run them, please download the datsets files form the given link below and place them inside their respective directories (see instructions for each dataset below...).  

<p style="text-align: center;">Samll Datasets</p>  
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

<p style="text-align: center;">Medium Datasets</p>  
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

<p style="text-align: center;">Large Datasets</p>  
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
    After downloading the dataset from the provided link, follow the instructions at https://www.hindawi.com/journals/bmri/2014/781670/#supplementary-materials to prepare it for analysis

## Dataset Preparation
### Variable P
For synthetic datasets, we varied the availability of each auxiliary input feature independently by a uniform distribution of probability $p$, i.e., each auxilairy feature is available for $100p\%$. For more information about this, follow paper - Aux-Net (https://link.springer.com/chapter/10.1007/978-3-031-30105-6_46)

## Files
<!-- To run the models, see
1. main.py: All the comparison models can be run from this.
2. baseline.py: To run the Baseline model (ODL)

The class definition for each comparison model is given in
 - AuxDrop.py

The class definition for ODL baseline is given in
 - ODL.py

The dataloader for each dataset is given in
 - dataset.py -->

## Control Parameters

<!-- For **main.py** file, 
1. `data_name`: "german", "svmguide3", "magic04", "a8a", "ItalyPowerDemand", "SUSY", "HIGGS"
2. `type`: "variable_p", "trapezoidal", "obsolete_sudden"
3. `model_to_run`: "AuxDrop_ODL", "AuxDrop_OGD", "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer", "AuxDrop_ODL_RandomAllLayer", "AuxDrop_ODL_RandomInAuxLayer", "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst"
4. `n`: Learning rate
5. `aux_feat_prob`: If `type = "variable_p"`, then `aux_feat_prob` needs to be defined. It is the availability of each auxiliary input feature independently by a uniform distribution of probability `aux_feat_prob`
6. `dropout_p`: The dropout value of AuxLayer
7. `max_num_hidden_layers`: Number of hidden layers
8. `qtd_neuron_per_hidden_layer`: Number of neurons in each hidden layers except the AuxLayer
9. `n_classes`: The number of output classes
10. `aux_layer`: The position of the AuxLayer in the architecture
11. `n_neuron_aux_layer`: Number of neurons in the AuxLayer
12. `b`: This is a parameter of ODL framework. It represents the discount rate
13. `s`: This is a parameter of ODL framework. It represents the smoothing rate -->

For **baseline.py** file,
<!-- 1. `data_name`: "SUSY", "HIGGS"
2. `model_to_run`: "ODL"
3. `data_type`: "only_base", "all_feat"
4. `n`: Learning rate
5. `max_num_hidden_layers`: Number of hidden layers
6. `qtd_neuron_per_hidden_layer`: Number of neurons in each hidden layers
7. `n_classes`: The number of output classes
8. `b`: It represents the discount rate
9. `s`: It represents the smoothing rate -->

## Dependencies
1. numpy
2. torch
3. pandas
4. random
5. tqdm
6. os
7. pickle
8. tdigest
9. statsmodels

## Running the code

<!-- To run the Aux-Drop model, change the control parameters accordingly in the **main.py** file and run
 - `python Code/main.py`

To run the baseline ODL model, change the control parameters accordingly in the **baseline.py** file and run
 - `python Code/baseline.py` -->