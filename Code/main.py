#--------------Libraries--------------#
import os
import argparse
import random
import numpy as np
import sys
sys.path.append('/code/DataCode/')

#--------------Import Functions--------------#
from Utils import utils, metric_utils
from DataCode import data_load
from Models.run_nb3 import run_nb3
from Models.run_fae import run_fae

#--------------All Variables--------------#
if __name__ == '__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=2023, type=int, help='Seeding Number')
    parser.add_argument('--type', default="noassumption", type=str,
                        choices = ["noassumption", "basefeatures", "bufferstorage"], 
                        help='The type of the experiment.')
    
    # Data Variables
    parser.add_argument('--dataname', default = "wpbc", type = str,
                        choices = ["imdb", "diabetes_us", "higgs", "susy", "a8a", "magic04", 
                                   "spambase", "krvskp", "svmguide3", "ipd", "german", 
                                   "diabetes_f", "wbc", "australian", "wdbc", "ionosphere", "wpbc"],
                        help='The name of the data')
    parser.add_argument('--syndatatype', default = "variable_p", type = str,
                        help = "The type to create suitable synthetic dataset")
    parser.add_argument('--probavailable', default = 0.75, type = float,
                        help = "The probability of each feature being available to create synthetic data")
    parser.add_argument('--ifbasefeat', default = False, type = bool,
                        help = "If base features are available")

    # Method Variables
    parser.add_argument('--methodname', default = "nb3", type = str,
                        choices = ["nb3", "fae", "olvf", "ocds", "ovfm", 
                                   "dynfo", "orf3v", "auxnet", "auxdrop"],
                        help = "The name of the method")
    parser.add_argument('--initialbuffer', default = 0, type = int,
                        help = "The storage size of initial buffer trainig")
    parser.add_argument('--ifimputation', default = False, type = bool,
                        help = "If some features needs to be imputed")
    parser.add_argument('--ifdummyfeat', default = False, type = bool,
                        help = "If some dummy features needs to be created")
    parser.add_argument('--ifAuxDropNoAssumpArchChange', default = False, type = bool,
                        help = "If the Aux-Drop architecture needs to be changed to handle no assumption")
    parser.add_argument('--nruns', default = 5, type =  int,
                        help = "The number of times a method should runs. For navie Bayes, it would be 1 because it is a deterministic method.")

    args = parser.parse_args()
    seed = args.seed
    type = args.type
    data_name = args.dataname
    syn_data_type = args.syndatatype
    p_available = args.probavailable
    if_base_feat = args.ifbasefeat
    method_name = args.methodname
    initial_buffer = args.initialbuffer
    if_imputation = args.ifimputation
    if_dummy_feat = args.ifdummyfeat
    if_auxdrop_no_assumption_arch_change = args.ifAuxDropNoAssumpArchChange
    n_runs = args.nruns
    
    data_type = "Synthetic"
    if data_name in ["imdb", "diabetes_us", "spamassasin", "naticusdroid", "crowdsense"]:
        data_type = "Real"
    #--------------SeedEverything--------------#
    utils.seed_everything(seed)

    #--------------Load Data wrt Variables--------------#
    if data_type == "Synthetic":
        X, Y, X_haphazard, mask = data_load.data_load_synthetic(data_name, syn_data_type, 
                                                p_available, if_base_feat)
    else:
        X, Y, X_haphazard, mask = data_load.data_load_real(data_name)
    
    #--------------Load Method wrt Variables--------------#
    if method_name == "nb3":
        # Model Config
        n_runs = 1 # NB3 is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
        numTopFeats_percent = [.2, .4, .6, .8, 1]
        X_haphazard = utils.prepare_data_naiveBayes(X, mask)
    if method_name == "fae":
        # Model Config - Based on original paper
        n_runs = 1 # NB3 is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
        m = 5    # (maturity) Number of instances needed before a learner’s classifications are used by the ensemble
        p = 3    # (probation time) is the number of times in a row a learner is allowed to be under the threshold before being removed
        f = 0.15 # (feature change threshold) is the threshold placed on the amount of change between the
                # youngest learner’s set of features (yfs) and the top M features (mfs);
        r = 10   # (growth rate) is the number of instances between when the last learner was added and
                # when the ensemble’s accuracy is checked for the addition of a new learner
        N = 50   # Number of instances over which to compute an accuracy measure;
        M = [.2, .4, .6, .8, 1]  # Number of features (here in proportion) selected by the feature selection algorithm for a newly created learner
        model_params_list = (m, p, f, r, N, M)
        X_haphazard = utils.prepare_data_naiveBayes(X, mask)

    #--------------Run Model--------------#
    if method_name == "nb3":
        results = run_nb3(X, X_haphazard, Y, numTopFeats_percent, n_runs)
    elif method_name == "fae":
        results = run_fae(X, Y, X_haphazard, n_runs, model_params_list)
    print(results)
    #--------------Calculate all Metrics--------------#


    #--------------Store results and all variables--------------#