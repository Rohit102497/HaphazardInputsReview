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

class Graph:
    def __init__(self, num_feats: int):
        self.G = np.random.random((num_feats, num_feats))        

    def retrive(self, X: np.array, X_mask: np.array) -> np.array:
        Xt = X*X_mask
        dt = sum(X_mask)
        X_rec = (1/dt) * np.dot(self.G, Xt)

        return X_rec

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
    
    def predict(self, X: np.array, X_rec: np.array, X_mask: np.array) -> float:
        obs_idx = np.where(X_mask)[0]
        rec_idx = np.where(X_mask == 0)[0]

        y_logit = self.p*np.dot(X[obs_idx], self.w[obs_idx]) + (1-self.p)*np.dot(X_rec[rec_idx], self.w[rec_idx])
        return y_logit.item()
    
    def loss(self, y: float, y_logit: float) -> np.float64:
        prod = y*y_logit
        if prod > np.log(sys.float_info.max):
            return -prod
        if prod < -np.log(sys.float_info.max):
            return 0.0
        return (1/np.log(2))*np.log1p(np.exp(-prod))
    
    def update(self, X: np.array, X_rec: np.array, X_mask: np.array, y: np.array):
        self.t+=1
        obs_idx = np.where(X_mask)[0]
        rec_idx = np.where(X_mask == 0)[0]

        self.L_observed += self.loss(y, np.dot(X[obs_idx], self.w[obs_idx]))
        self.L_reconstructed += self.loss(y, np.dot(X_rec[rec_idx], self.w[rec_idx]))

        if self.t == self.T:

            self.p = np.exp(-self.ita*self.L_observed)/(np.exp(-self.ita*self.L_observed) + np.exp(-self.ita*self.L_reconstructed))

            self.t=0
            self.L_observed=0
            self.L_reconstructed=0
    
    def sparcify(self):
        # truncate the min element by B
        min_idx = np.argmin(self.w)
        self.w[min_idx] *= self.gamma

class OCDS:
    def __init__(self, num_feats: int, T: int, gamma: float, alpha: float, beta0: float, beta1: float, beta2: float):
        # T     : number of steps after which 'p' is updated
        # 'p'   : weighing factor between observer-feature predictions and reconstructed-feature prediction
        # gamma : Sparcity factor
        self.alpha = alpha
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.t=0
        self.Graph = Graph(num_feats)
        self.Classifier = Classifier(num_feats, T, gamma)

    def predict(self, X: np.array, X_mask: np.array, X_rec: np.array=None) -> tuple[np.array]:
        if X_rec is None:
            X_rec = self.Graph.retrive(X, X_mask)
        y_logit = self.Classifier.predict(X, X_rec, X_mask)
        y_pred = int(y_logit>0)

        return y_pred, y_logit

    def update(self, X: np.array, X_mask: np.array, X_rec: np.array, Y: np.array):
        Y = -1 if Y==0 else Y.item()
        self.Classifier.update(X, X_rec, X_mask, Y)

        xt = X*X_mask*self.beta0
        dt = sum(X_mask)
        psi_xt = self.Graph.retrive(X, X_mask)*self.beta0
        wt = self.Classifier.w
        L = laplacian(self.Graph.G)
        It = np.eye(*self.Graph.G.shape)

        self.t+=1
        tau = np.sqrt(1/self.t)

        delta_w_F_term1 = -2*(Y - np.dot(wt,psi_xt))*psi_xt
        delta_w_F_term2 = self.beta1*entry_wise_subdifferential(wt)
        delta_w_F_term3 = self.beta2*np.dot((L + L.T), wt)

        delta_G_F_term1 = (-2/dt)*(Y - np.dot(wt,psi_xt))*It.T*np.outer(xt, wt)
        delta_G_F_term2 = (2*self.alpha/dt) * It.T * xt.reshape(-1, 1) * (xt - psi_xt*X_mask) * It

        delta_w_F = delta_w_F_term1 + delta_w_F_term2 + delta_w_F_term3
        delta_G_F = delta_G_F_term1 - delta_G_F_term2

        # Update classifier weights
        self.Classifier.w -= tau*delta_w_F
        # Update Graph edges
        self.Graph.G -= tau*delta_G_F

        # Sparcify classifier weights
        self.Classifier.sparcify()

    def partial_fit(self, X: np.array, X_mask: np.array, Y: np.array) -> tuple[np.array]:
        X_rec = self.Graph.retrive(X, X_mask)
        y_pred, y_logit = self.predict(X, X_mask, X_rec)
        self.update(X, X_mask, X_rec, Y)
        return y_pred, y_logit