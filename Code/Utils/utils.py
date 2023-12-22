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
def dummy_feat(n_instances, n_feat):
    return np.ones((n_instances, n_feat))
