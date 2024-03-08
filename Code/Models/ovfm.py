# This script is a wrapper on the code of OVFM provided in the source paper by the authors

import numpy as np

from Models.OVFM.source.em.online_expectation_maximization import OnlineExpectationMaximization
from Models.OVFM.source.onlinelearning.ftrl_adp import FTRL_ADP
from Models.OVFM.source.onlinelearning.ensemble import sigmoid

class OVFM:

    def __init__(self, decay_choice, contribute_error_rate, n_feat, all_cont_indices, all_ord_indices,
                 WINDOW_SIZE, decay_coef_change=False, batch_size_denominator=None, batch_c = 8):
        
        # decay_choice - decay update rules choices (see original code)
        # contribute_error_rate - used in the original code implementation of classifiers
        # n_feat - Total number of features (for ease of code)
        # all_cont_indices - Index of features containing continious type value
        # all_ord_indices - Index of features containing ordinal/discrete type value
        # NOTE: By default, all features are assumed to be continious, unill input otherwise
        # WINDOW_SIZE - a (buffer-like) window to store data instances
        # decay_coef_change - set ’True’ for learning rate decay, ’False’ otherwise
        # batch_size_denominator - used in update step in case of learning rate decay
        # batch_c -added to the denominator for stability in learning rate decay

        self.decay_choice = decay_choice
        self.contribute_error_rate = contribute_error_rate
        self.indices = list(range(n_feat+1))

        self.decay_coef_change = decay_coef_change
        self.batch_size_denominator = batch_size_denominator
        self.batch_c = batch_c
        self.j=0

        self.x_loss=0
        self.z_loss=0
        self.lamda=0.5
        self.eta = 0.001

        self.oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
        self.classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=n_feat+1)
        self.classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=n_feat+1)
    
    def partial_fit(self, X, Y):

        if len(X.shape) == 1:
            X.reshape(1, -1)
        
        if self.decay_coef_change:
            this_decay_coef = self.batch_c / (self.j/(self.batch_size_denominator*2) + self.batch_c)
        else:
            this_decay_coef = 0.5
        
        self.j += 1
        
        z_i, _ = self.oem.partial_fit_and_predict(X, max_workers=1,decay_coef=this_decay_coef)

        X = np.nan_to_num(X, nan=0)
        x_i = np.hstack(([[1]], X)).reshape(-1)
        z_i = np.hstack(([[1]], z_i)).reshape(-1)

        p_x, decay_x,loss_x, w_x = self.classifier_X.fit(self.indices, x_i, Y ,self.decay_choice, self.contribute_error_rate)
        p_z, decay_z,loss_z, w_z = self.classifier_Z.fit(self.indices, z_i, Y ,self.decay_choice, self.contribute_error_rate)

        p=sigmoid(self.lamda*np.dot(w_x,x_i)+(1.0-self.lamda)*np.dot(w_z,z_i))

        self.x_loss+=loss_x
        self.z_loss+=loss_z
        denominator = np.exp(-self.eta*self.x_loss)+np.exp(-self.eta*self.z_loss)
        if denominator == 0:
            self.lamda = self.lamda
        else:
            self.lamda=(np.exp(-self.eta*self.x_loss)/denominator)[0]

        y_logit = p
        y_pred = int(p>0.5)

        return y_pred, y_logit