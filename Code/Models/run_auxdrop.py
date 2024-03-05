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
    for k in range(len(model_params_list)):
        model_params = model_params_list[k]
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

def arch_change_create_param_list(model_params):
    params_list = []
    for max_num_hidden_layers in model_params['max_num_hidden_layers']:
        for qtd_neuron_per_hidden_layer in model_params['qtd_neuron_per_hidden_layer']:
            for b in model_params['b']:
                for s in model_params['s']:
                    for n in model_params['n']:
                        for dropout_p in model_params['dropout_p']:
                            params_list.append({'max_num_hidden_layers': max_num_hidden_layers,
                                'qtd_neuron_per_hidden_layer': qtd_neuron_per_hidden_layer,
                                'b': b, 's': s, 'n': n, 'dropout_p': dropout_p, 
                                'use_cuda': model_params['use_cuda'],
                                'n_aux_feat': model_params['n_aux_feat'], 
                                'batch_size': model_params['batch_size'],
                                'n_classes': model_params['n_classes'],
                                'n_neuron_aux_layer': model_params['n_neuron_aux_layer']
                            })
    return params_list

def run_auxdrop_arch_change(Y, X_haphazard, mask, num_runs, model_params):
    result = {}
    params_list = arch_change_create_param_list(model_params)
    print("Number of experiments to run: ", len(params_list))
    for k in range(len(params_list)):
        params = params_list[k]
        print("Experiment number: ", k+1, "\nParams: \n", params)
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []
            n_aux_feat = params["n_aux_feat"]
            start_time = time.time()
            model = AuxDrop_ODL_AuxLayer1stlayer(params["max_num_hidden_layers"], 
                        params["qtd_neuron_per_hidden_layer"], params["n_classes"],
                        params["n_neuron_aux_layer"], params["batch_size"], params["b"],
                        params["n"], params["s"], params["dropout_p"], params["n_aux_feat"],
                        params["use_cuda"])
            for i in tqdm(range(0, X_haphazard.shape[0])):
                model.partial_fit(X_haphazard[i].reshape(1, n_aux_feat),
                                mask[i].reshape(1,n_aux_feat), Y[i].reshape(1))
            for i in model.prediction:
                Y_logits.append(i[0, 1].item())
                Y_pred.append(torch.argmax(i).item())
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
            print("Run number: ", j+1, "\n Metrics: \n", eval_list[j])
        result[str(params)] = eval_list
    return result