from Models.auxdrop import AuxDrop_ODL, AuxDrop_ODL_AuxLayer1stlayer
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import time


def run_auxdrop(X_base, X_aux_new, aux_mask, Y, num_runs, model_params_list):
    result = {}
    for i in range(len(model_params_list)):
        model_params = model_params_list[0]
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []
            n_base_feat = X_base.shape[1]
            n_aux_feat = X_aux_new.shape[1]
            start_time = time.time()
            model = AuxDrop_ODL(*model_params)
            for i in tqdm(range(0, X_base.shape[0])):
                model.partial_fit(X_base[i].reshape(1,n_base_feat), X_aux_new[i].reshape(1, n_aux_feat),
                                   aux_mask[i].reshape(1,n_aux_feat), Y[i].reshape(1))
            for i in model.prediction:
                Y_logits.append(i[0, 1].item())
                Y_pred.append(torch.argmax(i).item())
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(model_params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result

def run_auxdrop_arch_change(Y, X_haphazard, mask, num_runs, model_params_list):
    result = {}
    for i in range(len(model_params_list)):
        model_params = model_params_list[0]
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []
            n_aux_feat = X_haphazard.shape[1]
            start_time = time.time()
            model = AuxDrop_ODL_AuxLayer1stlayer(*model_params)
            for i in tqdm(range(0, X_haphazard.shape[0])):
                model.partial_fit(X_haphazard[i].reshape(1, n_aux_feat),
                                mask[i].reshape(1,n_aux_feat), Y[i].reshape(1))
            for i in model.prediction:
                Y_logits.append(i[0, 1].item())
                Y_pred.append(torch.argmax(i).item())
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(model_params)] = eval_list
    return result