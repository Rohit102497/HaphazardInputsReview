import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, balanced_accuracy_score

# Number of Errors
def NumberOfErrors(y_actual, y_pred): # Two arrays: actual label and prediction
    return np.sum(y_actual != y_pred)

# accuracy
def Accuracy(y_actual, y_pred): # Two arrays: actual label and prediction
    return (np.sum(y_actual == y_pred)/y_actual.shape[0])*100

# auroc
def AUROC(y_actual, pred_logits):
    if np.sum(np.isnan(pred_logits)) == 0:
        return roc_auc_score(y_actual, pred_logits)*100
    else:
        return np.nan

# Balanced Accuracy
def BalancedAccuracy(y_actual, y_pred):
    return balanced_accuracy_score(y_actual, y_pred)*100

# AUPRC
def AUPRC(y_actual, pred_logits):
    if np.sum(np.isnan(pred_logits)) == 0:
        precision_val, recall_val, _ = precision_recall_curve(y_actual, pred_logits)
        return auc(recall_val, precision_val)*100
    else:
        return np.nan

def get_all_metrics(Y_true, Y_pred, Y_logits, time_taken):
    num_errors = NumberOfErrors(Y_true, Y_pred)
    accuracy = Accuracy(Y_true, Y_pred)
    auroc = AUROC(Y_true, Y_logits)
    auprc = AUPRC(Y_true, Y_logits)
    balanced_accuracy = BalancedAccuracy(Y_true, Y_pred)

    return {"Num. Errors"   : num_errors,
            "Accuracy"      : accuracy,
            "AUROC"         : auroc,
            "AUPRC"         : auprc,
            "Bal. Accuracy" : balanced_accuracy,
            "Time"          : time_taken}