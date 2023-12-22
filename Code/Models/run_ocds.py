from Models.ocds import OCDS
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def run_ocds(X, Y, X_haphazard, mask, num_runs, model_params_list):
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
            model = OCDS(*model_params)
            for i in tqdm(range(0, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
                # print(y_logit, y_pred)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(model_params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result