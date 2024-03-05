from Models.auxnet import AuxNet
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
import time

def create_param_list(model_params):
    params_list = []
    for bl in model_params['no_of_base_layers']:
        for el in model_params['no_of_end_layers']:
            for n in model_params['nodes_in_each_layer']:
                for b in model_params['b']:
                    for s in model_params['s']:
                            for lr in model_params['lr']:
                                params_list.append({'no_of_base_layers': bl, 'no_of_end_layers': el, 
                                    'nodes_in_each_layer': n, 'b': b, 's': s, 'lr': lr})
    return params_list

def run_auxnet(X_base, X, X_mask, Y, num_runs, model_params):
    result = {}
    params_list = create_param_list(model_params)
    print("number of experiments:", len(params_list))
    for k in range(len(params_list)): # Different combination of params
        params = params_list[k]
        eval_list = []
        print("Experiment number: ", k+1, "\nParams: \n", params)
        print("number of runs:", num_runs)
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []
            Y_true = Y
            Y = Y.astype(int).squeeze()
            Y_one_hot = np.zeros((Y.size, max(Y)+1))
            Y_one_hot[np.arange(Y.size), Y] = 1

            start_time = time.time()
            print("Input shape: ", X_base.shape[1])
            model = AuxNet(params['no_of_base_layers'], X.shape[1], params['no_of_end_layers'], 
                           params['nodes_in_each_layer'], params['b'], params['s'], X_base.shape[1])
            loss_fn = CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=params['lr'])
            # model = OLVF(params['C'], params['C_bar'], params['B'], params['reg'], params['n_feat0'])
            
            model.train()
            for i in tqdm(range(0, len(Y))):
                x_base = torch.tensor(X_base[i], dtype=torch.float).reshape(-1, 1)
                x = torch.tensor(X[i], dtype=torch.float).reshape(-1, 1)
                mask = torch.tensor(X_mask[i]).reshape(-1)
                y = torch.tensor(Y_one_hot[i], dtype=torch.float)
                
                optimizer.zero_grad()
                
                y_logit, y_logits = model(x, mask, x_base)
                
                losses_per_layer = []
                for logit in y_logits:
                    loss = loss_fn(logit, y)
                    losses_per_layer.append(loss)
                    loss.backward(retain_graph=True)
                
                optimizer.step()
                model.update_layer_weights(losses_per_layer, mask)
            
                
                y_pred = np.argmax(y_logit.detach().numpy())
                Y_pred.append(y_pred)
                Y_logits.append(y_logit[1].detach().item())
            taken_time = time.time() - start_time
            del model
            # print("Y:",  Y_true)
            # print("Y_pred:", Y_pred)
            # print("Y_logits:", Y_logits)
            # print(Y_true.shape, Y_pred.shape, Y_logits.shape)
            eval_list.append(get_all_metrics(np.array(Y_true).reshape(-1, 1), np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
            print("Run number: ", j+1, "\n Metrics: \n", eval_list[j])
        result[str(params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result
