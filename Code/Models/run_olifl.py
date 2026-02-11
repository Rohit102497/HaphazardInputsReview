from Models.olifl import OLIFL
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    params_list = []
    for C in model_params['C']:
        for option in model_params['option']:
            params_list.append({'C': C, 'option': option})
    return params_list

def run_olifl(X, Y, X_haphazard, mask, num_runs, model_params):
    result = {}
    params_list = create_param_list(model_params)
    print("Number of experiments: ", len(params_list))
    for k in range(len(params_list)):  # len(params_list)
        print("Experiment number: ", k+1)
        params = params_list[k]
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)

            start_time = time.time()
            params['seed'] = j
            model = OLIFL(params['C'], params['option'], params['seed'])
            Y_pred, Y_logits = model.fit(X, Y, X_haphazard, mask)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result