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

# Prepare data for naive BAyes
def process_single_input(x, mask):
    return {f"{i}": val for i, val in enumerate(x) if mask[i]}

def prepare_data_naiveBayes(X, mask):
    return list(map(process_single_input, X, mask))

# Dummy feature creation for aux-drop code
def dummy_feat(n_instances, n_feat, dummy_type = "standardnormal"):
    if dummy_type == "standardnormal":
        return np.random.normal(0, 1, (n_instances, n_feat))


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

