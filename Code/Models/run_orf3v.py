from Models.orf3v import ORF3V
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    params_list = []
    for forestSize in model_params['forestSize']:
        for replacementInterval in model_params['replacementInterval']:
            for replacementChance in model_params['replacementChance']:
                for windowSize in model_params['windowSize']:
                    for updateStrategy in model_params['updateStrategy']:
                        for alpha in model_params['alpha']:
                            for delta in model_params['delta']:
                                params_list.append({'forestSize': forestSize, 
                                                'replacementInterval': replacementInterval, 
                                                'replacementChance': replacementChance, 
                                                'windowSize': windowSize,
                                                'updateStrategy': updateStrategy,
                                                'alpha': alpha,
                                                'delta': delta
                                                })
    return params_list

def run_orf3v(X, Y, X_haphazard, mask, num_runs, model_params, initial_buffer):
    result = {}
    params_list = create_param_list(model_params)
    print("Num of experiments to run:", len(params_list))
    for k in range(len(params_list)): # 
        params = params_list[k]
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []

            startIdx = 1 if initial_buffer==0 else initial_buffer
            start_time = time.time()
            model = ORF3V(X_haphazard[:startIdx], mask[:startIdx], Y[:startIdx], params["forestSize"], params["replacementInterval"], 
                          params["replacementChance"], params["windowSize"], params["updateStrategy"], params["alpha"], params["delta"])
            for i in tqdm(range(startIdx, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y[startIdx:], np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result