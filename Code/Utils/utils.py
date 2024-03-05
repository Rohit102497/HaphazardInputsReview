import random
import os
import numpy as np
import torch

#--------------Seed--------------#
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Prepare data for naive Bayes
def process_single_input(x, mask):
    return {f"{i}": val for i, val in enumerate(x) if mask[i]}

def prepare_data_naiveBayes(X, mask):
    return list(map(process_single_input, X, mask))


# Impute to create base features
def impute_data(data, mask, imputation_type):
    if imputation_type == 'forwardfill':
        arr = data.T
        mask = np.invert(mask.astype(bool))
        mask = mask.T
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx,axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:,None], idx]
        # If there is no information to impute (for the initial instances), impute by 0
        return out.T
    elif imputation_type == 'forwardmean':
        with np.errstate(divide='ignore',invalid='ignore'):
            forward_mean_arr = np.cumsum(data, axis = 0)/np.cumsum(mask, axis = 0)
        forward_mean_arr[np.isnan(forward_mean_arr)] = 0
        mask = np.invert(mask.astype(bool))
        data[mask] = forward_mean_arr[mask]
        return data
    elif imputation_type == 'zerofill':
        return data
    return data

# Prepare data for AuxNet (can also be used for AuxDrop)
def prepare_data_imputation(X_haphazard, mask, imputation_type, n_impute_feat):
    base_index_list = []
    i = 0
    while n_impute_feat > len(base_index_list):
            temp_index = np.where(mask[i] == 1)[0]
            i  = i + 1
            base_index_list = list(set(base_index_list) | set(temp_index))
    
    base_index_list = base_index_list[:n_impute_feat]
    aux_index_list = [a for a in range(mask.shape[1]) if a not in base_index_list]
    X_base = impute_data(X_haphazard[:, base_index_list], mask[:, base_index_list], imputation_type)

    return X_base, X_haphazard[:, aux_index_list], mask[:, aux_index_list]


# Dummy feature creation for aux-drop and aux-net code
def dummy_feat(n_instances, n_feat, dummy_type = "standardnormal"):
    if dummy_type == "standardnormal":
        return np.random.normal(0, 1, (n_instances, n_feat))

