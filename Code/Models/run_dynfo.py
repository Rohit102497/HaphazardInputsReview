from Models.dynfo import DynFo
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    params_list = []
    for beta in model_params['beta']:
        for alpha in model_params['alpha']:
            for delta in model_params['delta']:
                for epsilon in model_params['epsilon']:
                    for gamma in model_params['gamma']:
                        for M in model_params['M']:
                            for theta1 in model_params['theta1']:
                                for theta2 in model_params['theta2']:
                                    params_list.append({'alpha': alpha, 'beta': beta, 
                                     'delta': delta, 'epsilon': epsilon, 'gamma': gamma,
                                     'M': M, 'theta1': theta1, 'theta2': theta2, 
                                     'N': model_params['N']})
    return params_list


def run_dynfo(X, Y, X_haphazard, mask, num_runs, model_params, initial_buffer):
    result = {}
    params_list = create_param_list(model_params)
    print("Number of experiments to run: ", len(params_list))
    for k in range(len(params_list)): # len(params_list)
        params = params_list[k]
        print("Experiment number: ", k+1, "\nParams: \n", params)
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []

            startIdx = 1 if initial_buffer==0 else initial_buffer
            start_time = time.time()
            model = DynFo(X_haphazard[:startIdx], mask[:startIdx], Y[:startIdx], 
                          params["alpha"], params["beta"], params["delta"], params["epsilon"],
                          params["gamma"], params["M"], params["N"], params["theta1"], 
                          params["theta2"])
            for i in tqdm(range(startIdx, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            # print("Model Weights: \n", model.weights, "\n Accepted Features: \n", model.acceptedFeatures)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y[startIdx:], np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
            print("Run number: ", j+1, "\n Metrics: \n", eval_list[j])
        result[str(params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result

