# Model Configs
import numpy as np
from Utils.utils import dummy_feat, impute_data


# --------- NB3 ------------
def config_nb3(data_name):
    '''
        numTopFeats_percent = [.2, .4, .6, .8, 1]
    '''
    # config_dict = {}
    # config_dict["numTopFeats_percent"] = numTopFeats_percent

    params_list = {
        'wpbc':         {"numTopFeats_percent":[1]},
        'ionosphere':   {"numTopFeats_percent":[0.2]},
        'wdbc':         {"numTopFeats_percent":[0.6]},
        'australian':   {"numTopFeats_percent":[0.8]},
        'wbc':          {"numTopFeats_percent":[0.2]},
        'diabetes_f':   {"numTopFeats_percent":[1]},
        'german':       {"numTopFeats_percent":[1]},
        'ipd':          {"numTopFeats_percent":[0.6]},
        'svmguide3':    {"numTopFeats_percent":[0.2]},
        'krvskp':       {"numTopFeats_percent":[1]},
        'spambase':     {"numTopFeats_percent":[1]},
        'magic04':      {"numTopFeats_percent":[0.6]},
        'a8a':          {"numTopFeats_percent":[0.2]},
        'susy':         {"numTopFeats_percent":[1]},
        'higgs':        {"numTopFeats_percent":[0.2]},
        'diabetes_us':  {"numTopFeats_percent":[0.8]},
        'spamassasin':  {"numTopFeats_percent":[0.2]},
        'imdb':         {"numTopFeats_percent":[0.4]},
        'crowdsense_c3':{"numTopFeats_percent":[0.6]},
        'crowdsense_c5':{"numTopFeats_percent":[0.8]},
    }
    
    n_runs = 1 # NB3 is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
    config_dict = params_list[data_name]
    return n_runs, config_dict

# --------- FAE ------------
def config_fae(data_name):
    # Based on original paper
    config_dict = {}
    n_runs = 1 # FAE is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
    m = 5    # (maturity) Number of instances needed before a learner’s classifications are used by the ensemble
    p = 3    # (probation time) is the number of times in a row a learner is allowed to be under the threshold before being removed
    f = 0.15 # (feature change threshold) is the threshold placed on the amount of change between the
            # youngest learner’s set of features (yfs) and the top M features (mfs);
    r = 10   # (growth rate) is the number of instances between when the last learner was added and
            # when the ensemble’s accuracy is checked for the addition of a new learner
    N = 50   # Number of instances over which to compute an accuracy measure;
    params_list = {
        'wpbc':         {"numTopFeats_percent":[0.4]},
        'ionosphere':   {"numTopFeats_percent":[0.2]},
        'wdbc':         {"numTopFeats_percent":[0.2]},
        'australian':   {"numTopFeats_percent":[1]},
        'wbc':          {"numTopFeats_percent":[0.8]},
        'diabetes_f':   {"numTopFeats_percent":[1]},
        'german':       {"numTopFeats_percent":[0.2]},
        'ipd':          {"numTopFeats_percent":[0.8]},
        'svmguide3':    {"numTopFeats_percent":[0.4]},
        'krvskp':       {"numTopFeats_percent":[1]},
        'spambase':     {"numTopFeats_percent":[0.2]},
        'magic04':      {"numTopFeats_percent":[1]},
        'a8a':          {"numTopFeats_percent":[0.2]},
        'susy':         {"numTopFeats_percent":[0.6]},
        'higgs':        {"numTopFeats_percent":[0.4]},
        'diabetes_us':  {"numTopFeats_percent":[0.4]},
        'spamassasin':  {"numTopFeats_percent":[0.2]},
        'imdb':         {"numTopFeats_percent":[0.8]},
        'crowdsense_c3':{"numTopFeats_percent":[0.2]},
        'crowdsense_c5':{"numTopFeats_percent":[0.8]},
    }
    # M = [.2, .4, .6, .8, 1]  # Number of features (here in proportion) selected by the feature selection algorithm for a newly created learner
    # if data_name == "higgs":
        # M = [.2, .4, .6] # For .8 it takes 31 hrs and for 1 it takes 202 hrs
    # Store all the config parameters
    config_dict["m"] = m
    config_dict["p"] = p
    config_dict["f"] = f
    config_dict["r"] = r
    config_dict["N"] = N
    config_dict["M"] = params_list[data_name]["numTopFeats_percent"]

    return n_runs, config_dict

# --------- OLVF ------------
def config_olvf(data_name, num_feat):
    n_runs = 1 # All w is 0. So it is deterministic
    '''
    Hyperparameter here means:

        'B':
        'C':
        'C_bar':
        'reg':
        'n_feat0':
    '''
    params_list = {
        'wpbc':         {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'ionosphere':   {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'wdbc':         {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'australian':   {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'wbc':          {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'diabetes_f':   {'B':[0.3],     'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'german':       {'B':[0.01],    'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'ipd':          {'B':[1],       'C':[1],           'C_bar':[0.01],     'reg':[0.0001]},
        'svmguide3':    {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'krvskp':       {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'spambase':     {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'magic04':      {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'imdb':         {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'a8a':          {'B':[1],       'C':[1],           'C_bar':[0.0001],   'reg':[0.0001]},
        'crowdsense_c3':{'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'crowdsense_c5':{'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'susy':         {'B':[1],       'C':[0.01],        'C_bar':[0.01],     'reg':[0.0001]},
        'higgs':        {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'diabetes_us':  {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'spamassasin':  {'B':[1],       'C':[1],           'C_bar':[0.0001],   'reg':[0.0001]},
    }
    # data_list_hyper = ['wbc', 'svmguide3', 'wpbc', 'ionosphere', 'magic04', 'german',
    #                     'spambase', 'wdbc', 'a8a']
    # data_list_hyper = []
    # config_dict = {}
    # if data_name in data_list_hyper:
    #     config_dict = params_list[data_name]
    # else:
    #     config_dict['B'] = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    #     config_dict['C_bar'] = [0.0001, 0.01, 1]
    #     config_dict['C'] = [0.0001, 0.01, 1]
    # config_dict['reg'] = [0.0001, 0.01, 1]
    
    config_dict = params_list[data_name]
    config_dict['n_feat0'] = num_feat
    
    return n_runs, config_dict

def config_ocds(num_feat, data_name):
    config_dict = {}
    gamma = [np.round(150/num_feat, 3)]# It is based on the number of features. The rule is to keep U_t < 150.
    if gamma[0] > 1:
        gamma = [1] # gamma cannot be more than 1

    params_list = {
        'wpbc':         {'T':[16], 'alpha':[0.001], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.0001], 'beta2':[0.01]},
        'ionosphere':   {'T':[8], 'alpha':[1], 'gamma':gamma, 'beta0':[1], 
                        'beta1': [0.0001], 'beta2':[0.01]},
        'wdbc':         {'T':[8], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [1], 'beta2':[0.0001]},
        'australian':   {'T':[16], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.01], 'beta2':[0.01]},
        'wbc':          {'T':[8], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [1], 'beta2':[0.01]},
        'diabetes_f':   {'T':[8], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.0001]},
        'crowdsense_c3':{'T':[8], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.01]},
        'crowdsense_c5':{'T':[8], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.01], 'beta2':[0.01]},
        'german':       {'T':[8], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [1], 'beta2':[0.0001]},
        'ipd':          {'T':[16], 'alpha':[1], 'gamma':gamma, 'beta0':[1], 
                        'beta1': [0.0001], 'beta2':[0.01]},
        'svmguide3':    {'T':[16], 'alpha':[1], 'gamma':gamma, 'beta0':[1], 
                        'beta1': [0.01], 'beta2':[0.0001]},
        'krvskp':       {'T':[16], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.01], 'beta2':[0.01]},
        'spambase':     {'T':[16], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.0001]},
        'spamassasin':  {'T':[16], 'alpha':[0.001], 'gamma':gamma, 'beta0':[0.001], 
                        'beta1': [0.001], 'beta2':[0.001]}, # Set heuristically
        'magic04':      {'T':[16], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.0001]},    
        'imdb':         {'T':[16], 'alpha':[1], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.01], 'beta2':[0.01]},  # Set heuristically
        'a8a':          {'T':[16], 'alpha':[1], 'gamma':gamma, 'beta0':[1], 
                        'beta1': [0.0001], 'beta2':[0.0001]},
        'diabetes_us':  {'T':[16], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [1], 'beta2':[0.0001]},
        'susy':         {'T':[16], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.0001]}, # Set heuristically
        'higgs':        {'T':[8], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.01], 'beta2':[0.0001]}, # Set heuristically
    }
    config_dict = params_list[data_name]

    # if data_name in ["imdb", "susy", "higgs", "spamassasin"]:
    #     config_dict = params_list[data_name]
    # else:
    #     T = [8, 16] # Update after T instances
    #     # gamma = [0.01, 0.1, 1] 
    #     alpha = [0.0001, 0.01, 1] # According to original paper
    #     # beta0 = [0.0000001] # We introduced this to absrob the magnitude of the first expression in equation 8
    #     beta0 = [0.0001, 0.01, 1] # We introduced this to absrob the magnitude of the first expression in equation 8
    #     beta1 = [0.0001, 0.01, 1] # According to original paper
    #     beta2 = [0.0001, 0.01, 1] # According to original paper
    #     config_dict['T'] = T
    #     config_dict['gamma'] = gamma
    #     config_dict['alpha'] = alpha
    #     config_dict['beta0'] = beta0
    #     config_dict['beta1'] = beta1
    #     config_dict['beta2'] = beta2

    
    return config_dict

# --------- OVFM ------------
def config_ovfm(data_name):
    config_dict = {}
    '''

    'decay_choice': Possible values - 0, 1, 2, 3, 4

    'contribute_error_rate': 

    'decay_coef_change': This is used to update decay coefficient (this_decay_coef). Set as False
    
    'batch_size_denominator': This is used to update decay coefficient (this_decay_coef). 
                              If 'decay_coef_change' is False, then the value of 'batch_size_denominator'
                              does not matter.
    
    Below the hyperparameters corresponding to different datasets are defined.

        Taken from Original Paper Code: ["ionosphere", "wdbc", "australian", 
                                        "diabetes_f", "german"]

        Heuristically Chosen: The following datasets requires significant time to run. Therefore,
                              we heuristically chose the hyperparameters based on the best performance
                              parameters of other similar size dataset.
                              ["imdb", "a8a", "susy", "higgs", "spamassasin", "diabetes_us"] 
        
        Hyperparameters Exhaustive Searching: Best hyperparameters for the following dataset were 
                              exhaustively search.
                              ["wpbc", "ipd", "svmguide3", "krvskp", "spambase", 
                               "magic04", "crowdsense_c3", "crowdsense_c5", ]
    '''
    params_dict = {
        "wpbc"          : {'decay_choice': [2], 'contribute_error_rate': [0.01],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "ionosphere"    : {'decay_choice': [4], 'contribute_error_rate': [0.02],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},                                     
        "wdbc"          : {'decay_choice': [0], 'contribute_error_rate': [0.02],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]}, 
        "australian"    : {'decay_choice': [4], 'contribute_error_rate': [0.01],
                            'decay_coef_change':[False] ,'batch_size_denominator': [10]},
        "wbc"           : {'decay_choice': [2], 'contribute_error_rate': [0.02],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]},
        "diabetes_f"    : {'decay_choice': [2], 'contribute_error_rate': [0.05],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]},
        "crowdsense_c3" : {'decay_choice': [4], 'contribute_error_rate': [0.01],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "crowdsense_c5" : {'decay_choice': [4], 'contribute_error_rate': [0.05],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},    
        "german"        : {'decay_choice': [3], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]},
        "ipd"           : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "svmguide3"     : {'decay_choice': [0], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "krvskp"        : {'decay_choice': [4], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "spambase"      : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "spamassasin"   : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "magic04"       : {'decay_choice': [3], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "imdb"          : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "a8a"           : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "diabetes_us"   : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "susy"          : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "higgs"         : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "default"       : {'decay_choice': [0, 1, 2, 3, 4], 'contribute_error_rate': [0.001, 0.005, 0.01, 0.02, 0.05],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},                                    
    }
    config_dict = params_dict[data_name]
    # if data_name not in ["ionosphere", "australian", "wbc", "diabetes_f", "german", 
    #                      "imdb", "a8a", "susy", "higgs", "spamassasin", "diabetes_us"]:
    #     config_dict = params_dict["default"]
    # else:
    #     config_dict = params_dict[data_name]
    
    return config_dict
    
# --------- DynFo ------------
def config_dynfo(num_of_instances, data_name):
    ''' Dynfo takes a lot of time to run, because at each instance, the model undergoes many 
    relearning operations. To make sure, that the model does not undergo many relearning 
    operation, we need to set higher beta and theta1 values.
    '''
    config_dict = {}
    # Setting the value of N as 10% of the data or 20 instances. Whichever is less
    N = int(num_of_instances*.1)
    if N > 20:
        N = 20

    ''' Original paper provides the best hyperparameter for imdb dataset. We only change 
        values of beta and theta1 such that it is feasible to run the experiment.'''
    params_list = {
        "wpbc":        {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "ionosphere":  {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "wdbc":        {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.8], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "australian":  {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.8], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "wbc":         {"alpha": [0.1], "beta": [0.5], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "diabetes_f":  {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "crowdsense_c3":{"alpha": [0.5], "beta": [0.5], "delta": [0.01], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "crowdsense_c5":{"alpha": [0.5], "beta": [0.5], "delta": [0.01], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "german":      {"alpha": [0.5], "beta": [0.5], "delta": [0.001], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "ipd":         {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "svmguide3":   {"alpha": [0.5], "beta": [0.5], "delta": [0.001], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "krvskp":      {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "spambase":    {"alpha": [0.1], "beta": [0.5], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.8], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "spamassasin": {"alpha": [0.5], "beta": [0.5], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "magic04":     {"alpha": [0.5], "beta": [0.5], "delta": [0.1], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "imdb":        {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "a8a":         {"alpha": [0.5], "beta": [0.5], "delta": [0.03], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "diabetes_us": {"alpha": [0.5], "beta": [0.5], "delta": [0.1], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},     
        "susy":        {"alpha": [0.5], "beta": [0.5], "delta": [0.4], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "higgs":       {"alpha": [0.5], "beta": [0.5], "delta": [0.2], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
                   
    }
    config_dict = params_list[data_name]

    # if data_name in ["imdb", "spamassasin", "diabetes_us", "magic04", "a8a", "susy", "higgs"]:
    #     config_dict = params_list[data_name]
    # else:
    #     alpha = [0.1, 0.5] # Lower Alpha value is good
    #     beta = [0.5, 0.8] # a lower value for dense data streams and a higher value for sparser data streams. 
    #     delta = [0.001, 0.01] # This should be small
    #     epsilon = [0.001, 0.01]  # Weak learners weights are decreased by epsilon value
    #     gamma = [0.5, 0.8]
    #     M = [500]
    #     theta1=[0.05] # This is set for each dataset as done in the original paper
    #     theta2=[0.75] # This is set for each dataset as done in the original paper

    #     # Store all the config parameters
    #     config_dict["alpha"] = alpha
    #     config_dict["beta"] = beta
    #     config_dict["delta"] = delta
    #     config_dict["epsilon"] = epsilon
    #     config_dict["gamma"] = gamma
    #     config_dict["M"] = M
    #     config_dict["N"] = N
    #     config_dict["theta1"] = theta1
    #     config_dict["theta2"] = theta2

    return config_dict

# --------- ORF3V ------------
def config_orf3v(data_name):
    # config_dict = {}
    # forestSize = [3, 5, 10] # Number of Stumps for every feature
    # replacementInterval = [5, 10] # Instances after which to replace stumps
    # replacementChance = [0.7] # If replacement strategy is random, then this is the probability not to replace each stump
    # windowSize = [20] # Buffer - stores instances on which to determine feature statistics
    # updateStrategy = ["oldest", "random"] # replacement strategy: "oldest", "random"
    # alpha = [0.01, 0.1, 0.3, 0.5, 0.9] # weight update parameter
    # delta = [0.001] # caculates hb, which is used for pruning.
    # if data_name in ["susy", "higgs", "imdb"]: # heuristically set
    #     forestSize = [5]
    #     replacementInterval = [10]
    #     updateStrategy = ["random"] # replacement strategy: "oldest", "random"
    #     alpha = [0.1] # weight update parameter
    # config_dict["forestSize"] = forestSize
    # config_dict["replacementInterval"] = replacementInterval
    # config_dict["replacementChance"] = replacementChance
    # config_dict["windowSize"] = windowSize
    # config_dict["updateStrategy"] = updateStrategy
    # config_dict["alpha"] = alpha
    # config_dict["delta"] = delta
        
    params_list = {
        "wpbc":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "ionosphere":  {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wdbc":        {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "australian":  {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wbc":         {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "diabetes_f":  {"forestSize": [3], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "crowdsense_c3":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "crowdsense_c5":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "german":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "ipd":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "svmguide3":   {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "krvskp":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "spambase":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "spamassasin": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "diabetes_us": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    config_dict = params_list[data_name]

    return config_dict

# --------- Aux-Net ------------
def config_auxnet(data_name):
    config_dict = {}

    params_list = {
        "wpbc":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "ionosphere":  {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "wdbc":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.01]},
        "australian":  {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.01]},
        "wbc":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "diabetes_f":  {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.05]},
        "crowdsense_c3":{"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.001]},
        "crowdsense_c5":{"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.001]},
        "german":      {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "ipd":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "svmguide3":   {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.1]},
        "krvskp":      {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "spambase":    {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.005]},
        "spamassasin": {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "magic04":     {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.5]},
        "imdb":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "a8a":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "diabetes_us": {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
        "susy":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
        "higgs":       {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
    }
    
    config_dict = params_list[data_name]

    return config_dict


# --------- Aux-Drop ------------
def config_auxdrop(if_auxdrop_no_assumption_arch_change, X, data_name,
                    if_imputation, if_dummy_feat, n_dummy_feat, X_haphazard,
                    mask, imputation_type, dummy_type):
    config_dict = {}
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
    
    # We are choosing approximately 5 times the number of features. Note that this is only for faster computation. We can also add as and when new features comes in.
    n_neuron_aux_layer_dict = {"australian": 100, "wbc": 100, "diabetes_f": 100, "german": 100,
                              "ipd": 100, "svmguide3": 100, "magic04": 100, "susy": 100, "higgs": 100,
                              "wpbc": 200, "ionosphere": 200, "wdbc": 200, "krvskp": 200, "diabetes_us": 200,  
                              "spambase": 300, "a8a": 500,
                              "crowdsense_c3": 5000, "crowdsense_c5": 5000,
                              "spamassasin": 30000, "imdb": 30000}
    # n_dict = {"wpbc": [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5], 
    # }

    max_num_hidden_layers = [6] # Number of hidden layers
    qtd_neuron_per_hidden_layer = [50] # Number of nodes in each hidden layer except the AuxLayer
    n_classes = 2 # The total number of classes (output labels)
    batch_size = 1 # The batch size is always 1 since it is based on stochastic gradient descent
    b = [0.99] # discount rate
    s = [0.2] # smoothing parameter

    # n = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5] # learning rate
    n = {"wpbc": [0.01], "ionosphere": [0.5], "wdbc": [0.01], "australian": [0.005],
         "wbc": [0.1], "diabetes_f": [0.001], "german": [0.001], "ipd": [0.3],
         "svmguide3": [0.001], "krvskp": [0.1], "spambase": [0.005], "magic04": [0.01],
         "a8a": [0.01], "susy": [0.05], "higgs": [0.05], 
         "imdb": [0.01], "crowdsense_c3": [0.001], "crowdsense_c5": [0.001],
         "diabetes_us": [0.05], "spamassasin": [0.01]}
    
    # dropout_p = [0.3, 0.5] # The dropout rate in the AuxLayer
    dropout_p = {"wpbc": [0.5], "ionosphere": [0.3], "wdbc": [0.5], "australian": [0.5],
         "wbc": [0.3], "diabetes_f": [0.5], "german": [0.5], "ipd": [0.3],
         "svmguide3": [0.3], "krvskp": [0.3], "spambase": [0.5], "magic04": [0.3],
         "a8a": [0.3], "susy": [0.3], "higgs": [0.3], 
         "imdb": [0.3], "crowdsense_c3": [0.5], "crowdsense_c5": [0.3],
         "diabetes_us": [0.3], "spamassasin": [0.3]}

    n_aux_feat = X.shape[1] # Number of auxiliary features
    n_neuron_aux_layer = n_neuron_aux_layer_dict[data_name] # The total numebr of neurons in the AuxLayer
    use_cuda = False
    config_dict["max_num_hidden_layers"] = max_num_hidden_layers
    config_dict["qtd_neuron_per_hidden_layer"] = qtd_neuron_per_hidden_layer
    config_dict["n_classes"] = n_classes
    config_dict["n_neuron_aux_layer"] = n_neuron_aux_layer
    config_dict["batch_size"] = batch_size
    config_dict["b"] = b
    config_dict["s"] = s
    config_dict["n"] = n[data_name]
    config_dict["dropout_p"] = dropout_p[data_name]
    config_dict["n_aux_feat"] = n_aux_feat
    config_dict["use_cuda"] = use_cuda
    if if_auxdrop_no_assumption_arch_change:
        return config_dict
    else:
        # features_size - Number of base features
        # aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1. 
        features_size =  2 # number of base features
        aux_layer = [3] # The position of auxiliary layer. This code does not work if the AuxLayer position is 1.
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
        config_dict["features_size"] = features_size
        config_dict["aux_layer"] = aux_layer
        config_dict["n_aux_feat"] = n_aux_feat
        return config_dict, X_base, X_aux_new, aux_mask