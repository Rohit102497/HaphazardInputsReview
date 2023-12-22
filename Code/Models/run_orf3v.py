from Models.orf3v import ORF3V
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def run_orf3v(X, Y, X_haphazard, mask, num_runs, model_params_list, initial_buffer):
    result = {}
    for i in range(len(model_params_list)):
        model_params = model_params_list[0]
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []

            startIdx = 1 if initial_buffer==0 else initial_buffer
            start_time = time.time()
            model = ORF3V(*model_params, X_haphazard[:startIdx], mask[:startIdx], Y[:startIdx])
            for i in tqdm(range(startIdx, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y[startIdx:], np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(model_params)] = eval_list
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result