from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from Models.nb3 import NB3
from tqdm import tqdm
import numpy as np
import time

# Model run
def run_nb3(X, X_haphazard, Y, numTopFeats_percent, num_runs):
    result = {}
    for i in range(len(numTopFeats_percent)):
        eval_list = []
        numTopFeats = round(X.shape[1]*numTopFeats_percent[i])
        no_instances = X.shape[0]
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)

            Y_pred = []
            Y_logits = []

            start_time = time.time()
            model = NB3()
            for x, y in tqdm(zip(X_haphazard, Y), total=no_instances):
                y_pred, y_logits = model.partial_fit(x, y, numTopFeats)
                Y_pred.append(y_pred)
                Y_logits.append(y_logits[1])
            taken_time = time.time() - start_time
            del model

            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[numTopFeats] = eval_list
    # The structure of results: It is dictionary with key being the number of features and value are the metrics.
    return result