from Models.olvf import OLVF
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    params_list = []
    for C in model_params['C']:
        for C_bar in model_params['C_bar']:
            for B in model_params['B']:
                for reg in model_params['reg']:
                    params_list.append({'C': C, 'C_bar': C_bar, 'B': B, 
                        'reg': reg, 'n_feat0': model_params['n_feat0']})
    return params_list

def run_olvf(X_haphazard, mask, Y, num_runs, model_params):
    result = {}
    params_list = create_param_list(model_params)
    print("number of runs:", num_runs)
    for k in range(len(params_list)): # Different combination of params
        params = params_list[k]
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []

            start_time = time.time()
            model = OLVF(params['C'], params['C_bar'], params['B'], params['reg'], params['n_feat0'])
            for i in tqdm(range(0, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result