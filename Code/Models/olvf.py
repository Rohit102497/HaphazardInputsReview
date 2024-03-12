import numpy as np

eps = np.finfo(float).eps

def sigmoid(x):
    return 1/(1+np.exp(x))

class FeatureSpaceClassifier:
    def __init__(self, C_bar, B, n_feat0):
        self.C_bar = C_bar
        self.B = B
        self.w_bar = np.zeros(n_feat0, dtype=float)
        # self.w_bar = np.random.uniform(0, 1, n_feat0)
    
    # predict projection confidences
    def predict(self, X_mask, seen_feature):
        R_w = np.zeros_like(self.w_bar)
        R_w[seen_feature] = 1.0
        R_x = X_mask
        Pw = sigmoid(np.dot(R_w, self.w_bar))
        Px = sigmoid(np.dot(R_x, self.w_bar))

        return Pw, Px
    
    # loss function for feature space classifier
    def loss(self, shared, new, Y, Y_pred, X_mask):
        if Y == -1:
            Y = 0
        I = float(int(np.squeeze(Y)) == int(np.squeeze(Y_pred)))

        exp1 = 1
        exp2 = 1
        if len(shared) != 0:
            ws = self.w_bar[shared]
            R_xs = X_mask[shared]
            exp1 = np.exp(-I * np.dot(ws, R_xs))
        if len(new) != 0:
            wn = self.w_bar[new]
            R_xn = X_mask[new]
            exp2 = np.exp(-I * np.dot(wn, R_xn))

        l =  np.log(1 + (exp1 * exp2))
        d_w_bar = np.zeros_like(self.w_bar) 
        if len(shared) != 0:
            d_w_bar[shared] = -1.0 * (np.log(exp1*exp2) / np.log(1 + exp1*exp2)) * I * R_xs
        if len(new) != 0:
            d_w_bar[new] = -1.0 * (np.log(exp1*exp2) / np.log(1 + exp1*exp2)) * I * R_xn

        return l, d_w_bar
    
    # Update parameters
    def update_params(self, shared, new, tau, d_w_bar):
        if len(shared) != 0:
            self.w_bar[shared] = self.w_bar[shared] + tau * d_w_bar[shared]
        if len(new) != 0:
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
    def sparcify(self, seen_features):
        remaining_feat_no = max(1, int(np.floor(self.B*len(seen_features))))
        if len(seen_features) > remaining_feat_no:
            idx_to_keep = np.argsort(self.w_bar[seen_features])[:-remaining_feat_no]
            m = np.zeros_like(self.w_bar[seen_features], dtype=float)
            m[idx_to_keep] = 1.0
            self.w_bar[seen_features] *= m

class InstanceClassifier:
    def __init__(self, C, B, reg, n_feat0):
        self.C = C
        self.B = B
        self.reg = reg
        # self.w = np.random.uniform(0, 1, n_feat0)
        self.w = np.zeros(n_feat0)
    
    # predict output from shared feature space
    def predict(self, shared, X):
        if len(shared) == 0: # There is nothing to predict on
            y_logit = np.random.normal(0, 1)
        else:
            xs, ws = X[shared], self.w[shared]
            y_logit = np.dot(xs, ws)
        y_pred =  float(y_logit>0)
        y_logit = abs(y_logit)
        return y_pred, y_logit
    
    # modified hinge loss for instance classifier
    def loss(self, shared, new, X, Y):

        if len(shared) != 0:
            ws = self.w[shared]
            xs = X[shared]
            ds = np.dot(xs, ws)
        if len(new) != 0:
            wn = self.w[new]
            xn = X[new]
            dn = np.dot(xn, wn)

        y = -1 if Y == 0 else Y

        if len(shared) == 0:
            return max(0, 1 - y*dn)
        if len(new) == 0:
            return max(0, 1 - y*ds)
        return max(0, 1 - y*ds - y*dn)

    # Update parameters
    def update_params(self, shared, new, X, Y, Pw, Px, tau):

        # print("w shared: ", self.w[shared], "Pw:", Pw, "tau:", tau, "Y:", Y, "x_shared:", X[shared])
        if len(shared) != 0:
            self.w[shared] = self.w[shared] + Pw * tau * Y * X[shared]
        if len(new) != 0:
            self.w[new] = Px * tau * Y * X[new]

        # print("updated w: ", self.w)
        return

    # Partial Update step
    def update(self, shared, new, X, Y, Pw, Px):

        l = self.loss(shared, new, X, Y)

        # In the paper this term is multiple below term is multiplied by -1. However, that does not make sense.
        # Because then the next term "tau = min(self.C, tau)" would always be tau and not self.C since tau is a negative value
        tau = l / (np.linalg.norm(X)**2 + eps) 
        tau = min(self.C, tau)

        self.update_params(shared, new, X, Y, Pw, Px, tau)
        return

    # Sparcity step
    def sparcify(self, w_bar, seen_features):
        factor = self.reg / (np.dot(w_bar, self.w) + eps)
        factor = min(1, factor)

        self.w *= factor

        remaining_feat_no = max(1, int(np.floor(self.B*len(seen_features))))
        if len(seen_features) > remaining_feat_no:
            idx_to_keep = np.argsort(self.w[seen_features])[:-remaining_feat_no]
            m = np.zeros_like(self.w[seen_features], dtype=float)
            m[idx_to_keep] = 1.0
            self.w[seen_features] *= m

class OLVF:
    def __init__(self, C, C_bar, B, reg, n_feat0):
        
        # C -  loss bounding parameter for instance classifier [.0001, .01, 1]
        # C_bar - loss bounding parameter for feature classifier [.0001, .01, 1]
        # B proportion of selected features for sparsity [.01, .1, .3, .5, .7, .9, 1]
        # reg - regularization parameter
        # n_feat0 - Total number of features to be ecountered (for ease of coding)
        
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

        # 1. Update FeatureSpace classifier

        self.featureSpaceClassifier.update(list(self.sharedFeatures), list(self.newFeatures), X_mask, Y, Y_pred)

        # 2. Get feature space confidences
        Pw, Px = self.featureSpaceClassifier.predict(X_mask, list(self.featureSet))

        # 3. Update InstanceClassifier using projection confidences
        self.instanceClassifier.update(list(self.sharedFeatures), list(self.newFeatures), X, Y, Pw, Px)

        # 4. Sparcity step for FeatureSpace classifier and Instance classifier
        self.instanceClassifier.sparcify(self.featureSpaceClassifier.w_bar, list(self.featureSet))
        self.featureSpaceClassifier.sparcify(list(self.featureSet))
        

    def partial_fit(self, X, X_mask, Y):
        if Y == 0:
            Y = -1
        y_pred, y_logit = self.predict(X, X_mask)

        self.update(X, X_mask, Y, y_pred)

        return y_pred, y_logit