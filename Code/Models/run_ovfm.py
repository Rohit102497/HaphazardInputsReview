from Models.ovfm import OVFM
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    params_list = []
    for decay_choice in model_params['decay_choice']:
        for contribute_error_rate in model_params['contribute_error_rate']:
            for decay_coef_change in model_params['decay_coef_change']:
                for batch_size_denominator in model_params['batch_size_denominator']:
                    params_list.append({'decay_choice': decay_choice, 
                                        'contribute_error_rate': contribute_error_rate, 
                                        'decay_coef_change': decay_coef_change, 
                                        'batch_size_denominator': model_params['batch_size_denominator']})
    return params_list


def get_cont_indices(X):
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices

def run_ovfm(X, Y, X_haphazard, mask, num_runs, model_params):
    result = {}
    params_list = create_param_list(model_params)
    print("number of runs:", len(params_list))
    for k in range(len(params_list)):
        params = params_list[k]
        eval_list = []
        print("Experiment number: ", k+1, "\nParams: \n", params)
        for j in range(num_runs):            
            # Seeding for model
            seed_everything(j)
            X_masked = np.ones_like(X)*np.nan
            X_masked[mask.astype(bool)] = X[mask.astype(bool)]

            all_cont_indices=get_cont_indices(X_masked)
            all_ord_indices=~all_cont_indices
            batch_c = 8

            n_feat = X_masked.shape[1]
            Y_true = Y.flatten()
            Y_pred = []
            Y_logits = []

            '''
            CHANGES DONE IN THIS CODE AS OPPOSED TO ORIGINAL

            max_iter: Since it is an online learning problem, we iter one by one and we only see data once and
                    not twice, as implemented in original code.
            
            BATCH_SIZE : This should be set to 1 as  we should process one data at a time.

            WINDOW_SIZE : Window basically serves as buffer in this model, and keeping in accordance with other models
                        we set it as minimum of 500 or 10% of total instances. This makes sure that the model has enough
                        buffer size and is consistent with the size of data i.e. if the number of instances in the data
                        is small, we have a smaller window size, and if the dataset is larges, we have a larger window size.
                        Capping of window size at 500 makes sure that the data storage used is reasonable.

            this_decay_coef: As value of 'j' differs from previous to new implementation, we changed the update
                            equation of 'this_decay_coef', to make sure it is consistent with previous implementation
                            the equation is changes form 'this_decay_coef = batch_c / (j + batch_c)' to
                            'this_decay_coef = batch_c / (j/(batch_size_denominator*2) + batch_c)'.
            '''
            # WINDOW_SIZE is the buffer size. Therefore we set it as the minimum of 10% of total instances of 500 instances.
            WINDOW_SIZE = min(20, int(X_masked.shape[0]*.1))

            start_time = time.time()
            model = OVFM(params['decay_choice'], params['contribute_error_rate'], n_feat,
                         all_cont_indices, all_ord_indices, WINDOW_SIZE,
                         params['decay_coef_change'], params['batch_size_denominator'], batch_c)
            
            for i in tqdm(range(len(Y_true))):
                x, y = X_masked[[i]], Y_true[[i]]
                y_pred, y_logit = model.partial_fit(x, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y_true.reshape(-1, 1), np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
            print("Run number: ", j+1, "\n Metrics: \n", eval_list[j])
        result[str(params)] = eval_list
    return result