from Models.ocds import OCDS
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    params_list = []
    for t in model_params['T']:
        for gamma in model_params['gamma']:
            for alpha in model_params['alpha']:
                for beta0 in model_params['beta0']:
                    for beta1 in model_params['beta1']:
                        for beta2 in model_params['beta2']:
                            params_list.append({'T': t, 'gamma': gamma, 'alpha': alpha, 
                                'beta0': beta0, 'beta1': beta1, 'beta2': beta2})
    return params_list

def run_ocds(X, Y, X_haphazard, mask, num_runs, model_params):
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
            Y_pred = []
            Y_logits = []

            start_time = time.time()
            model = OCDS(X.shape[1], params['T'], params['gamma'], params['alpha'], 
                         params['beta0'], params['beta1'], params['beta2'])
            for i in tqdm(range(0, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
                # print(y_logit, y_pred)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result