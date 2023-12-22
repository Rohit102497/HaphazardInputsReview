import numpy as np

eps = np.finfo(float).eps

def sigmoid(x):
    return 1/(1+np.exp(x))

class FeatureSpaceClassifier:
    def __init__(self, C_bar, B, n_feat0):
        self.C_bar = C_bar
        self.B = B
        self.w_bar = np.zeros(n_feat0, dtype=float)
    
    # predict projection confidences
    def predict(self, X_mask):
        R_w = np.ones_like(self.w_bar)
        R_x = X_mask
        Pw = sigmoid(np.dot(R_w, self.w_bar))
        Px = sigmoid(np.dot(R_x, self.w_bar))

        return Pw, Px
    
    # loss function for feature space classifier
    def loss(self, shared, new, Y, Y_pred, X_mask):
        I = float(int(Y) == int(Y_pred))

        ws = self.w_bar[shared]
        wn = self.w_bar[new]

        R_xs = X_mask[shared]
        R_xn = X_mask[new]

        exp1 = np.exp(-I * np.dot(ws, R_xs))
        exp2 = np.exp(-I * np.dot(wn, R_xn))

        l =  (1 - (exp1 * exp2))

        dl_dws = -1.0 * (np.log(exp1*exp2) / np.log(1 + exp1*exp2)) * I * R_xs
        dl_dwn = -1.0 * (np.log(exp1*exp2) / np.log(1 + exp1*exp2)) * I * R_xn

        d_w_bar = np.zeros_like(self.w_bar)
        d_w_bar[shared] = dl_dws
        d_w_bar[new] = dl_dwn

        return l, d_w_bar
    
    # Update parameters
    def update_params(self, shared, new, tau, d_w_bar):
        # self.w_bar[existing] = self.w_bar[existing]
        self.w_bar[shared] = self.w_bar[shared] + tau * d_w_bar[shared]
        self.w_bar[new] = - tau * d_w_bar[new]
        return
    
    # Calculate loss and update weights
    def update(self, shared, new, X_mask, Y, Y_pred):

        l, d_w_bar = self.loss(shared, new, Y, Y_pred, X_mask)
        denominator = np.linalg.norm(d_w_bar[shared])**2 + np.linalg.norm(d_w_bar[new])**2
        if denominator == 0:
            tau = self.C_bar
        else:
            tau = min(self.C_bar, -1.0 * (l / denominator))
        self.update_params(shared, new, tau, d_w_bar)
        return
    
    # Sparcity step
    def sparcify(self):
        # truncate the min element by B
        min_idx = np.argmin(self.w_bar)
        self.w_bar[min_idx] *= self.B

class InstanceClassifier:
    def __init__(self, C, B, reg, n_feat0):
        self.C = C
        self.B = B
        self.reg = reg
        self.w = np.zeros(n_feat0)
    
    # predict output from shared feature space
    def predict(self, shared, X):
        xs, ws = X[shared], self.w[shared]

        y_logit = np.dot(xs, ws)
        y_pred =  float(y_logit>0)

        y_logit = abs(y_logit)
        return y_pred, y_logit
    
    # modified hinge loss for instance classifier
    def loss(self, shared, new, X, Y):

        ws = self.w[shared]
        xs = X[shared]

        wn = self.w[new]
        xn = X[new]

        ds = np.dot(xs, ws)
        dn = np.dot(xn, wn)

        y = -1 if Y == 0 else Y

        return max(0, 1 - y*ds - y*dn)

    # Update parameters
    def update_params(self, shared, new, X, Y, Pw, Px, tau):

        self.w[shared] = self.w[shared] + Pw * tau * Y * X[shared]
        self.w[new] = - Px * tau * Y * X[new]
        return

    # Partial Update step
    def update(self, shared, new, X, Y, Pw, Px):

        l = self.loss(shared, new, X, Y)

        tau = -1.0 * l / (np.linalg.norm(X)**2 + eps)
        tau = min(self.C, tau)

        self.update_params(shared, new, X, Y, Pw, Px, tau)
        return

    # Sparcity step
    def sparcify(self, w_bar):
        
        factor = self.reg / (np.dot(w_bar, self.w) + eps)
        factor = min(1, factor)

        self.w *= factor

        # truncate the min element by B
        min_idx = np.argmin(self.w)
        self.w[min_idx] *= self.B

class OLVF:
    def __init__(self, C, C_bar, B, reg, n_feat0):
        self.C = C
        self.C_bar = C_bar
        self.B = B

        self.featureSet = set()

        self.sharedFeatures = set()
        self.newFeatures = set()

        self.featureSpaceClassifier = FeatureSpaceClassifier(C_bar, B, n_feat0)
        self.instanceClassifier = InstanceClassifier(C, B, reg, n_feat0)
    
    def predict(self, X, X_mask):

        currentFeatures = set(np.where(X_mask)[0])

        self.sharedFeatures = self.featureSet.intersection(currentFeatures)
        self.newFeatures = currentFeatures.difference(self.featureSet)

        self.featureSet.update(currentFeatures)

        y_pred, y_logit = self.instanceClassifier.predict(list(self.sharedFeatures), X)
        return y_pred, y_logit

    def update(self, X, X_mask, Y, Y_pred):

        # 1. Get feature space confidences
        Pw, Px = self.featureSpaceClassifier.predict(X_mask)

        # 2. Update FeatureSpace classifier
        self.featureSpaceClassifier.update(list(self.sharedFeatures), list(self.newFeatures), X_mask, Y, Y_pred)

        # 3. Update InstanceClassifier using projection confidences
        self.instanceClassifier.update(list(self.sharedFeatures), list(self.newFeatures), X, Y, Pw, Px)

        # 4. Sparcity step for FeatureSpace classifier and Instance classifier
        self.featureSpaceClassifier.sparcify()
        self.instanceClassifier.sparcify(self.featureSpaceClassifier.w_bar)

    def partial_fit(self, X, X_mask, Y):

        y_pred, y_logit = self.predict(X, X_mask)

        self.update(X, X_mask, Y, y_pred)

        return y_pred, y_logit