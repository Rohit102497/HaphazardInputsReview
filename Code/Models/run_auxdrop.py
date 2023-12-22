from Models.auxdrop import AuxDrop_ODL
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

            start_time = time.time()
            model = AuxDrop_ODL(*model_params)
            for i in tqdm(range(0, X_base.shape[0]))):
                model.partial_fit(X_base[i].reshape(1,n_base_feat), X_aux_new[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1,n_aux_feat), Y[i].reshape(1))
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            taken_time = time.time() - start_time
            prediction = []
            for i in model.prediction:
                prediction.append(torch.argmax(i).item())
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(model_params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result