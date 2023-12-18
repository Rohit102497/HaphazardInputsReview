import random
import os
import numpy as np

#--------------Seed--------------#
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# Prepare data for naive BAyes
def process_single_input(x, mask):
    return {f"{i}": val for i, val in enumerate(x) if mask[i]}

def prepare_data_naiveBayes(X, mask):
    return list(map(process_single_input, X, mask))
