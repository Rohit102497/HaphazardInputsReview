import sys
import warnings
import numpy as np
from scipy.sparse.csgraph import laplacian

# It is not explained in the paper how a entry-wise subdifferential operator functions,
# and we did not find any substancial explainations on the internet either
# so we implement an entry-wise subdifferential operator as per our understanding
def entry_wise_subdifferential(w):
    return np.sign(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_sigmoid(x):
    return -np.logaddexp(0, -x)

# Custom implementation of a graph representing the strength of correlation between 2 features
class Graph:
    def __init__(self, num_feats: int):
        self.G = np.random.random((num_feats, num_feats))        

    def retrive(self, X: np.array, X_mask: np.array, Ut: list, reduce: bool=False) -> np.array:
        Xt = X*X_mask
        dt = sum(X_mask)
        X_rec = (1/dt) * np.dot(self.G, Xt)

        if reduce:
            return X_rec[Ut]
        
        U_mask = np.zeros_like(X_mask)
        U_mask[Ut] = 1.0
        return X_rec*U_mask

class Classifier:
    def __init__(self, num_feats: int, T: int, gamma: float):
        self.w = np.zeros(num_feats)
        self.T = T                          # number of steps after which 'self.p' is updated
        self.gamma = gamma                  # Sparcity factor
        self.t = 0                          # to keep track of number of steps
        self.ita = 8*np.sqrt(1/np.log(T))   # tuned parameter (as mentioned in paper)
        self.p=0.5                          # weighing parameter between observed features and reconstructed features
        self.L_observed=0                   # Accumulated loss from observed feature prediction
        self.L_reconstructed=0              # Accumulated loss from reconstructed feature prediction
    
    def predict(self, X: np.array, X_rec: np.array, curr_feats:set, Ut:set) -> float:
        obs_idx = list(curr_feats)
        rec_idx = list(Ut.difference(curr_feats))
        y_logit = self.p*np.dot(X[obs_idx], self.w[obs_idx]) + (1-self.p)*np.dot(X_rec[rec_idx], self.w[rec_idx])
        return y_logit.item()
    
    def loss(self, y: float, y_logit: float) -> np.float64:
        prod = y*y_logit
        if prod > np.log(sys.float_info.max):
            return 0.0
        if prod < -np.log(sys.float_info.max):
            return (1/np.log(2))*(-prod)
        return (1/np.log(2))*np.log1p(np.exp(-prod))
    
    def update(self, X: np.array, X_rec: np.array, y: np.array, curr_feats: set, Ut: set):
        self.t+=1
        obs_idx = list(curr_feats)
        rec_idx = list(Ut.difference(curr_feats))
        self.L_observed += self.loss(y, np.dot(X[obs_idx], self.w[obs_idx]))
        self.L_reconstructed += self.loss(y, np.dot(X_rec[rec_idx], self.w[rec_idx]))
        if self.t == self.T:
            num = np.exp(-self.ita*self.L_observed)
            den = (np.exp(-self.ita*self.L_observed) + np.exp(-self.ita*self.L_reconstructed))
            if den == 0:
                self.p = 0.5 # Both Loss are very high and hence p is equal for both loss
            else:
                self.p = num/den

            self.t=0
            self.L_observed=0
            self.L_reconstructed=0
    
    def sparcify(self, seen_features):
        remaining_feat_no = max(1, int(np.floor(self.gamma*len(seen_features))))
        if len(seen_features) > remaining_feat_no:
            idx_to_keep = np.argsort(self.w[seen_features])[:-remaining_feat_no]
            m = np.zeros_like(self.w[seen_features], dtype=float)
            m[idx_to_keep] = 1.0
            self.w[seen_features] *= m

class OCDS:
    def __init__(self, num_feats: int, T: int, gamma: float, alpha: float, beta0: float, beta1: float, beta2: float):
        
        # num_feats - Total number of features that can be observed (for ease of coding)
        # T - Number of instances after which 'p' (weighing factor) is updated
        # gamma - Sparcity factor
        # alpha - Absorption scale parameter used in equation 10 of paper
        # beta0 - absorption scale introduced by us for the 1st term in eq. 9 of paper
        # beta1 - tradeoff parameter used in equation 9 of paper
        # beta2 - tradeoff parameter used in equation 9 of paper
                                                     
        self.Ut = set()
        self.curr_features = set()
        self.alpha = alpha
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.t=0
        self.Graph = Graph(num_feats)
        self.Classifier = Classifier(num_feats, T, gamma)

    def predict(self, X: np.array, X_mask: np.array, X_rec: np.array=None) -> tuple[np.array]:
        if X_rec is None:
            self.curr_features = set(np.where(X_mask)[0])
            self.Ut.update(self.curr_features)
            X_rec = self.Graph.retrive(X, X_mask, list(self.Ut), reduce=False)
        
        y_logit = self.Classifier.predict(X, X_rec, self.curr_features, self.Ut)
        y_pred = int(y_logit>0)

        return y_pred, y_logit

    def update(self, X: np.array, X_mask: np.array, X_rec: np.array, Y: np.array):
        # Performs updates according to the update rules mentioned in the paper
        Y = -1 if Y==0 else Y.item()
        self.Classifier.update(X, X_rec, Y, self.curr_features, self.Ut)

        xt = X[(list(self.curr_features))]*self.beta0
        dt = sum(X_mask)

        psi_xt = self.Graph.retrive(X, X_mask, list(self.Ut), reduce=True)*self.beta0
        pi_psi_xt = self.Graph.retrive(X, X_mask, list(self.Ut), reduce=False)[list(self.curr_features)]*self.beta0
        wt = self.Classifier.w[list(self.Ut)]

        ixgrid1 = np.ix_(list(self.Ut), list(self.Ut))
        ixgrid2 = np.ix_(list(self.curr_features), list(self.Ut))
        It = np.eye(*self.Graph.G.shape)

        G = self.Graph.G[ixgrid1]       # Ut x Ut
        L = laplacian(G)                # Ut x Ut
        It = It[ixgrid2]                # dt x Ut

        xt = xt.reshape(-1, 1)          # dt x 1
        psi_xt = psi_xt.reshape(-1, 1)  # Ut x 1
        pi_psi_xt = pi_psi_xt.reshape(-1, 1) # dt x 1
        wt = wt.reshape(-1, 1)          # Ut x 1

        self.t+=1
        tau = np.sqrt(1/self.t)
        delta_w_F_term1 = -2*(Y - wt.T@psi_xt)*psi_xt
        delta_w_F_term2 = self.beta1*entry_wise_subdifferential(wt)
        delta_w_F_term3 = self.beta2*(L + L.T)@wt

        delta_G_F_term1 = (-2/dt)*(Y - wt.T@psi_xt) * It.T@xt@wt.T
        delta_G_F_term2 = (2*self.alpha/dt) * It.T@xt@(xt - pi_psi_xt).T@It

        delta_w_F = delta_w_F_term1 + delta_w_F_term2 + delta_w_F_term3
        delta_G_F = delta_G_F_term1 - delta_G_F_term2

        # Update classifier weights
        self.Classifier.w[list(self.Ut)] = self.Classifier.w[list(self.Ut)] - tau*np.squeeze(delta_w_F)
        # Update Graph edges
        self.Graph.G[ixgrid1] -= tau*delta_G_F

        # Sparcify classifier weights
        self.Classifier.sparcify(list(self.Ut))

    def partial_fit(self, X: np.array, X_mask: np.array, Y: np.array) -> tuple[np.array]:
        self.curr_features = np.where(X_mask)[0]
        self.Ut.update(self.curr_features)
        X_rec = self.Graph.retrive(X, X_mask, list(self.Ut), reduce=False)
        y_pred, y_logit = self.predict(X, X_mask, X_rec)
        self.update(X, X_mask, X_rec, Y)
        return y_pred, y_logit