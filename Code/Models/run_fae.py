from Models.fae import FAE
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time


def run_fae(X, Y, X_haphazard, num_runs, params_list, data_name):
    M_list = params_list["M"] # Number of features (here in proportion) selected by the feature selection algorithm for a newly created learner
    result = {}
    for i in range(len(M_list)):
        eval_list = []
        numTopFeats = round(X.shape[1]*M_list[i])
        model_params = (params_list["m"], params_list["p"], 
                        params_list["f"], params_list["r"], 
                        params_list["N"], numTopFeats)
        no_instances = X.shape[0]
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []

            start_time = time.time()
            model = FAE(*model_params, data_name=data_name, Document=None, DocClass=None)
            for x, y in tqdm(zip(X_haphazard, Y), total=no_instances):
                y_pred, y_logit = model.partial_fit(x, int(np.squeeze(y)))
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[M_list[i]] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result

