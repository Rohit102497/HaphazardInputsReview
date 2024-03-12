# Libraries required
import numpy as np

from DataCode.data_utils import data_load_wpbc, data_load_wdbc, data_load_wbc, data_load_ionosphere
from DataCode.data_utils import data_load_australian, data_load_german, data_load_ipd, data_load_svmguide3
from DataCode.data_utils import data_load_krvskp, data_load_spambase, data_load_magic04, data_load_a8a
from DataCode.data_utils import data_load_susy, data_load_higgs, data_load_diabetes_f
from DataCode.data_utils import data_load_diabetes_us, data_load_imdb, data_load_spamassasin
from DataCode.data_utils import data_load_crowdsense_c3, data_load_crowdsense_c5

# Making sure that each instance has atleast one feature
def check_mask_each_instance(mask):
    index_0 = np.where(np.sum(mask, axis = 1) == 0)
    random_index = np.random.randint(mask.shape[1], size = (len(index_0[0])))
    # print(mask.shape, index_0, len(index_0[0]))
    for i in range(len(index_0[0])):
        mask[index_0[0][i], random_index[i]] = 1
    return mask


def data_load_synthetic(data_folder = "wpbc", type = "variable_p", p_available = 0.75, if_base_feat = False):

    if data_folder == "wpbc":
        X, Y = data_load_wpbc(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]
        
        # Masking
        # Note that the 33rd feature has missing values in 4 cases. 
        # We consider this as unavailable and include this info in masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask[:,32][np.where(np.isnan(X[:,32]))] = 0
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
    
    if data_folder == "ionosphere":
        X, Y = data_load_ionosphere(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "wdbc":
        X, Y = data_load_wdbc(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask

    if data_folder == "australian":
        X, Y = data_load_australian(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask

    if data_folder == "wbc":
        X, Y = data_load_wbc(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]
        
        # Masking
        # Note that the 6th feature has missing values in 16 cases. 
        # We consider this as unavailable and include this info in masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask[:,5][np.where(np.isnan(X[:,5]))] = 0
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask

    if data_folder == "diabetes_f":
        X, Y = data_load_diabetes_f(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
            
    if data_folder == "german":
        X, Y = data_load_german(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "ipd":
        X, Y = data_load_ipd(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
    
    if data_folder == "svmguide3":
        X, Y = data_load_svmguide3(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "krvskp":
        X, Y = data_load_krvskp(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "spambase":
        X, Y = data_load_spambase(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "magic04":
        X, Y = data_load_magic04(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "a8a":
        X, Y = data_load_a8a(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "susy":
        X, Y = data_load_susy(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
    if data_folder == "higgs":
        X, Y = data_load_higgs(data_folder)    
        n_feat = X.shape[1]
        number_of_instances = X.shape[0]

        # Masking
        if type == "variable_p" and not if_base_feat:
            mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
            mask = check_mask_each_instance(mask)
            X_haphazard = np.where(mask, X, 0)
            return X, Y, X_haphazard, mask
        
def data_load_real(data_folder = "diabetes_us"):
    if data_folder == "diabetes_us":
        X, Y = data_load_diabetes_us(data_folder)    
        mask = np.ones((X.shape))
        mask[np.isnan(X)] = 0
        X_haphazard = np.where(mask, X, 0)
        return X, Y, X_haphazard, mask
    
    if data_folder == "imdb":
        X, Y = data_load_imdb(data_folder)    
        mask = np.ones((X.shape))
        mask[np.isnan(X)] = 0
        X_haphazard = np.where(mask, X, 0)
        return X, Y, X_haphazard, mask
    
    if data_folder == "spamassasin":
        X, Y = data_load_spamassasin(data_folder)    
        mask = np.ones((X.shape))
        mask[np.isnan(X)] = 0
        X_haphazard = np.where(mask, X, 0)
        return X, Y, X_haphazard, mask
    
    if data_folder == "crowdsense_c3":
        X, Y = data_load_crowdsense_c3(data_folder)    
        mask = np.ones((X.shape))
        mask[np.isnan(X)] = 0
        X_haphazard = np.where(mask, X, 0)
        return X, Y, X_haphazard, mask

    if data_folder == "crowdsense_c5":
        X, Y = data_load_crowdsense_c5(data_folder)    
        mask = np.ones((X.shape))
        mask[np.isnan(X)] = 0
        X_haphazard = np.where(mask, X, 0)
        return X, Y, X_haphazard, mask
    