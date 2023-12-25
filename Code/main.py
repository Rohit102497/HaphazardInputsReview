#--------------Libraries--------------#
import os
import argparse
import random
import numpy as np
import sys
sys.path.append('/code/DataCode/')

#--------------Import Functions--------------#
from Utils import utils, metric_utils
from Utils.utils import dummy_feat, impute_data
from DataCode import data_load
from Models.run_nb3 import run_nb3
from Models.run_fae import run_fae
from Models.run_olvf import run_olvf
from Models.run_dynfo import run_dynfo
from Models.run_orf3v import run_orf3v
from Models.run_ocds import run_ocds
from Models.run_auxdrop import run_auxdrop, run_auxdrop_arch_change
from Models.run_ovfm import run_ovfm

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
    parser.add_argument('--imputationtype', default = 'forwardfill', type = str,
                        choices = ['forwardfill', 'forwardmean', 'zerofill'],
                        help = "The type of imputation technique to create base features")
    parser.add_argument('--ifdummyfeat', default = False, type = bool,
                        help = "If some dummy features needs to be created")
    parser.add_argument('--dummytype', default = 'standardnormal', type = str,
                        help = "The type of technique to create dummy base features")
    parser.add_argument('--ndummyfeat', default = 1, type = int,
                        help = "The number of dummy features to create")    
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
    imputation_type = args.imputationtype
    if_dummy_feat = args.ifdummyfeat
    dummy_type = args.dummytype
    n_dummy_feat = args.ndummyfeat
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
    
    #--------------Model Configs--------------#
    if method_name == "nb3":
        # Model Config
        n_runs = 1 # NB3 is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
        numTopFeats_percent = [.2, .4, .6, .8, 1]
        X_haphazard = utils.prepare_data_naiveBayes(X, mask)
    elif method_name == "fae":
        # Model Config - Based on original paper
        n_runs = 1 # FAE is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
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
    elif method_name == "olvf":
        n_runs = 1
        params_list = {
        'wbc':          {'C':0.6,    'C_bar':1,      'B':0.01},
        'svmguide3':    {'C':0.4,    'C_bar':0.1,    'B':0.001},
        'wpbc':         {'C':0.7,    'C_bar':0.0001, 'B':0.01},
        'ionosphere':   {'C':0.1,    'C_bar':0.01,   'B':0.1},
        'magic04':      {'C':0.6,    'C_bar':0.1,    'B':0.0001},
        'german':       {'C':0.5,    'C_bar':0.01,   'B':0.001},
        'spambase':     {'C':0.3,    'C_bar':0.01,   'B':0.001},
        'wdbc':         {'C':0.9,    'C_bar':1,      'B':0.1},
        'a8a':          {'C':0.05,   'C_bar':0.001,  'B':0.00001},
        'imdb':         {'C':0.01,   'C_bar':0.00001,  'B':0.01}
        }
        data_list_hyper = ['wbc', 'svmguide3', 'wpbc', 'ionosphere', 'magic04', 'german',
                           'spambase', 'wdbc', 'a8a', 'imdb']
        params = {}
        if data_name in data_list_hyper:
            params = params_list[data_name]
        else:
            params['C'] = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            params['C_bar'] = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
            params['B'] = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        params['n_feat0'] = X.shape[1]
        params['reg'] = 0.01
        default = {'C':0.55,   'C_bar':0.01,  'B':0.01}
    elif method_name == "dynfo":
        # Model Config
        alpha = 0.5
        beta = 0.3
        delta = 0.01
        epsilon = 0.001 
        gamma = 0.7
        M = 1000
        N = 1000
        theta1=0.05
        theta2=0.6
        model_params_list = [(alpha, beta, delta, epsilon, gamma, M, N, theta1, theta2)]
    elif method_name == "orf3v":
        # Model Config
        forestSize = 50
        replacementInterval = 5
        replacementChance = 0.7
        windowSize = 200
        updateStrategy = "random" # "oldest", "random"
        alpha = 0.01
        delta = 0.01
        model_params_list = [(forestSize, replacementInterval, replacementChance, windowSize, 
                              updateStrategy, alpha, delta)]
    elif method_name == "ocds":
        # Model Config
        num_feats = X.shape[1]
        T = 10
        gamma = 0.01
        alpha = 0.01
        beta0 = 0.0001 # 1, 0.1, 0.001, 0.0001, 0.00001, 0.000001
        beta1 = 0.01
        beta2 = 0.01
        model_params_list = [(num_feats, T, gamma, alpha, beta0, beta1, beta2)]
    elif method_name == "auxdrop":
        if if_auxdrop_no_assumption_arch_change:
            # max_num_hidden_layers - Number of hidden layers
            # qtd_neuron_per_hidden_layer - Number of nodes in each hidden layer except the AuxLayer
            # n_classes - The total number of classes (output labels)
            # n_neuron_aux_layer - The total numebr of neurons in the AuxLayer
            # batch_size - The batch size is always 1 since it is based on stochastic gradient descent
            # b - discount rate
            # n - learning rate
            # s - smoothing rate
            # dropout_p - The dropout rate in the AuxLayer
            # n_aux_feat - Number of auxiliary features
            # aux_feat_prob - The probability of each auxiliary feature being available at each point in time
            max_num_hidden_layers = 6 # Number of hidden layers
            qtd_neuron_per_hidden_layer = 50 # Number of nodes in each hidden layer except the AuxLayer
            n_classes = 2 # The total number of classes (output labels)
            n_neuron_aux_layer = 200 # The total numebr of neurons in the AuxLayer
            batch_size = 1 # The batch size is always 1 since it is based on stochastic gradient descent
            b = 0.99 # discount rate
            s = 0.2 # learning rate
            n = 0.1 # smoothing rate
            dropout_p = 0.5 # The dropout rate in the AuxLayer
            n_aux_feat = X.shape[1] # Number of auxiliary features
            use_cuda = False
            model_params_list = [(max_num_hidden_layers, qtd_neuron_per_hidden_layer,
                    n_classes, n_neuron_aux_layer, batch_size, b, n, s, 
                    dropout_p, n_aux_feat, use_cuda)]
        else:
            # features_size - Number of base features
            # max_num_hidden_layers - Number of hidden layers
            # qtd_neuron_per_hidden_layer - Number of nodes in each hidden layer except the AuxLayer
            # n_classes - The total number of classes (output labels)
            # aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1. 
            # n_neuron_aux_layer - The total numebr of neurons in the AuxLayer
            # batch_size - The batch size is always 1 since it is based on stochastic gradient descent
            # b - discount rate
            # n - learning rate
            # s - smoothing rate
            # dropout_p - The dropout rate in the AuxLayer
            # n_aux_feat - Number of auxiliary features
            # aux_feat_prob - The probability of each auxiliary feature being available at each point in time
            features_size =  2 # number of base features
            max_num_hidden_layers = 6 # Number of hidden layers
            qtd_neuron_per_hidden_layer = 50 # Number of nodes in each hidden layer except the AuxLayer
            n_classes = 2 # The total number of classes (output labels)
            aux_layer = 3 # The position of auxiliary layer. This code does not work if the AuxLayer position is 1.
            n_neuron_aux_layer = 200 # The total numebr of neurons in the AuxLayer
            batch_size = 1 # The batch size is always 1 since it is based on stochastic gradient descent
            b = 0.99 # discount rate
            s = 0.2 # learning rate
            n = 0.1 # smoothing rate
            dropout_p = 0.5 # The dropout rate in the AuxLayer
            n_aux_feat = X.shape[1] # Number of auxiliary features
            use_cuda = False
            if if_imputation:
                n_aux_feat = X.shape[1] - features_size # We impute some features (feature_size) to create base features. 
                    # Therefore number of base features would be total number of features - total number of base features
                # Create dataset
                X_base = impute_data(X_haphazard[:, :features_size],
                                    mask[:, :features_size], imputation_type)
                X_aux_new = X_haphazard[:, features_size:]
                aux_mask = mask[:, features_size:]
            elif if_dummy_feat:
                features_size = n_dummy_feat # We create dummy feature as base feature
                # Create dataset
                X_base = dummy_feat(X_haphazard.shape[0], features_size, dummy_type)
                X_aux_new = X_haphazard
                aux_mask = mask
            model_params_list = [(features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer,
                    n_classes, aux_layer, n_neuron_aux_layer, batch_size, b, n, s, 
                    dropout_p, n_aux_feat, use_cuda)]
    elif method_name == "ovfm":
        # Model Config
        c = 0.01 # Model sparsificity
        all_cont = False # Check which features are continous and ordinal
        lr = .01 # Learning rate
        B = 30 # Buffer Size. To calculate mean and standard deviation
        model_params_list = [(c, all_cont, lr, B)]

    #--------------Run Model--------------#
    result = {}
    if method_name == "nb3":
        results = run_nb3(X, X_haphazard, Y, numTopFeats_percent, n_runs)
    elif method_name == "fae":
        results = run_fae(X, Y, X_haphazard, n_runs, model_params_list)
    elif method_name == "olvf":
        result = run_olvf(X_haphazard, mask, Y, n_runs, params)
    elif method_name == "dynfo":
        result = run_dynfo(X, Y, X_haphazard, mask, n_runs, model_params_list, initial_buffer)
    elif method_name == "orf3v":
        result = run_orf3v(X, Y, X_haphazard, mask, n_runs, model_params_list, initial_buffer)
    elif method_name == "ocds":
        result = run_ocds(X, Y, X_haphazard, mask, n_runs, model_params_list)
    elif method_name == "auxdrop":
        if if_auxdrop_no_assumption_arch_change:
            result = run_auxdrop_arch_change(Y, X_haphazard, mask, n_runs, model_params_list)
        else:
            result = run_auxdrop(X_base, X_aux_new, aux_mask, Y, n_runs, model_params_list)
    elif method_name == "ovfm":
        result = run_ovfm(X, Y, X_haphazard, mask, n_runs, model_params_list, initial_buffer)
    print(result)

    #--------------Calculate all Metrics--------------#


    #--------------Store results and all variables--------------#