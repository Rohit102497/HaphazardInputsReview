import random
import numpy as np

class Queue:
    def __init__(self, n: int):
        self.queue = []
        self.n = n
    
    def push(self, instance: any):
        self.queue.append(instance)
        if len(self.queue) > self.n:
            self.queue.pop(0)

class InstanceBuffer:
    def __init__(self, n: int):
        self.X = Queue(n)
        self.X_mask = Queue(n)
        self.Y = Queue(n)
    
    def add(self, X: np.array, X_mask: np.array, Y: np.array):
        self.X.push(X)
        self.X_mask.push(X_mask)
        self.Y.push(Y)
    
    def get(self, feature: int) -> tuple[np.array]:
        X_mask = np.array(self.X_mask.queue)
        X = np.array(self.X.queue)
        Y = np.array(self.Y.queue)

        instances = np.where(X_mask[:,feature])[0]
        return X[instances, feature], Y[instances]

class DecisionStump:
    def __init__(self):
        self.best_gini = float('inf')
        self.accepted_features = set()
        self.feature_index = None
        self.threshold = None
        self.prediction_positive = None
        self.prediction_negative = None

    def fit(self, X: np.array, y: np.array, feature_index: int):
        y = y.reshape(-1)
        self.accepted_features.add(feature_index)

        unique_values = set(X)
        for threshold in unique_values:
            mask = X <= threshold
            y_positive = y[mask]
            y_negative = y[~mask]

            gini = self.calculate_gini(y_positive, y_negative)

            if gini < self.best_gini:
                self.best_gini = gini
                self.feature_index = feature_index
                self.threshold = threshold
                self.prediction_positive = self.get_majority_class(y_positive)
                self.prediction_negative = self.get_majority_class(y_negative)

    def predict(self, X: np.array) -> int:
        return self.prediction_positive if X[self.feature_index] <= self.threshold else self.prediction_negative

    # def predict_logit(self, X: np.array) -> float:
    #     prediction = self.predict(X)
    #     logit = 1.0 if prediction == self.prediction_positive else -1.0
    #     return logit
    
    # def predict_proba(self, X: np.array) -> float:
    #     # Calculate probabilities using a sigmoid function
    #     logit = self.predict_logit(X)
    #     probability = 1 / (1 + np.exp(-logit))
    #     return probability

    def calculate_gini(self, y_positive: np.array, y_negative: np.array) -> float:
        total_samples = len(y_positive) + len(y_negative)
        gini_positive = 1.0 - sum((np.sum(y_positive == c) / len(y_positive)) ** 2 for c in set(y_positive))
        gini_negative = 1.0 - sum((np.sum(y_negative == c) / len(y_negative)) ** 2 for c in set(y_negative))
        gini = (len(y_positive) / total_samples) * gini_positive + (len(y_negative) / total_samples) * gini_negative
        return gini

    def get_majority_class(self, y: list) -> int:
        unique_labels = set(y)
        if not unique_labels:
            return 0
        return int(max(unique_labels, key=y.tolist().count))

    def splitDescision(self) -> int:
        return self.feature_index

class Window(Queue):
    def __init__(self, n):
        super().__init__(n)
    
    def add(self, accuracy):
        self.push(accuracy)
    
    def errorRate(self):
        return sum(self.queue)/len(self.queue)

class DynFo:
    def __init__(self, alpha = 0.5, beta = 0.3, delta = 0.01, epsilon = 0.001, 
                gamma = 0.7, M = 1000, N = 1000, theta1=0.05, theta2=0.6, 
                Xs: np.array=None, X_masks: np.array=None, Ys: np.array=None,
                num_classes = 2):
        # α = 0.5, β = 0.3, δ = 0.01, epsilon = 0.001, γ = 0.7, θ1 = 0.05, θ2 = 0.6, M = 1000, N = 1000 on real world dataset
        self.alpha      = alpha
        self.beta       = beta
        self.delta      = delta
        self.epsilon    = epsilon
        self.gamma      = gamma
        self.M          = M
        self.N          = N
        self.theta1     = theta1
        self.theta2     = theta2

        self.num_classes = num_classes

        self.FeatureSet = set()
        self.window = Window(self.N)

        self.weights = [1] * self.M
        self.acceptedFeatures = []
        self.currentFeatures = set()
        self.learners = []

        self.instance_buffer = InstanceBuffer(self.N)

        if Xs is not None and X_masks is not None and Ys is not None:
            self.initialUpdate(Xs, X_masks, Ys)

    def initialUpdate(self, Xs, X_masks, Ys):
        
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)
            X_masks = X_masks.reshape(1, -1)
            Ys = Ys.reshape(1, -1)

        for X, X_mask, Y in zip(Xs, X_masks, Ys):
            # Initialize the FeatureSet using the first instance
            self.updateFeatureSet(X_mask)
            self.instance_buffer.add(X, X_mask, Y)

        # Initialize M learners (using the first instance)
        for _ in range(self.M):
            self.initNewLearner()
    
    def updateFeatureSet(self, X_mask):
        self.currentFeatures = set(np.where(X_mask)[0])
        self.FeatureSet.update(self.currentFeatures)

    def initNewLearner(self):

        # Choose 'accepted features' from featureSet for new learner, using the delta parameter
        numFeatures = max(int(self.delta*len(self.FeatureSet)), 1)
        accepted_features = random.sample(sorted(self.FeatureSet), numFeatures)

        # Get instances from the instance buffer that has the accepted features, and train a new learner on those instances
        for feature in accepted_features:
            X, Y = self.instance_buffer.get(feature)
            newLearner = DecisionStump()
            newLearner.fit(X, Y, feature)

        # Add new trained Learner to ensemble
        self.learners.append(newLearner)
        self.acceptedFeatures.append(accepted_features)
        self.weights.append(1)

    def updateGamma(self):
        # NOT FOUND IN PAPER
        # Retains the constant value of gamma
        self.gamma = self.gamma
        
    def relearnLearner(self, i):
        # Choose 'accepted features' from featureSet, using the delta parameter
        numFeatures = max(int(self.delta*len(self.FeatureSet)), 1)
        accepted_features = random.sample(sorted(self.FeatureSet), numFeatures)

        # Get instances from the instance buffer that has the accepted features, and re-train learner on those instances
        for feature in accepted_features:
            X, Y = self.instance_buffer.get(feature)
            learner = self.learners[i]
            learner.fit(X, Y, feature)

    def dropLearner(self, i):
        self.learners.pop(i)
        self.weights.pop(i)
        self.acceptedFeatures.pop(i)
        # Sanity Check:
        assert (len(self.weights) == len(self.learners)) and (len(self.acceptedFeatures) == len(self.learners))
    
    def predict(self, X):
        wc = np.array([0.0] * self.num_classes)
        for learner, weight in zip(self.learners, self.weights):
            if learner.splitDescision() in self.currentFeatures:
                pred = learner.predict(X)
                wc[pred] += weight
        
        return np.argmax(wc), max(wc)

    def update(self, X, Y):

        for learner, weight in zip(self.learners, self.weights):
            if learner.splitDescision() not in self.currentFeatures:
                weight -= self.epsilon
            else:
                i = int(learner.predict(X) == Y)
                weight = (i*2*self.alpha + weight) / (1 + self.alpha)
        
        q2 = np.quantile(self.weights, self.theta2)
        q1 = np.quantile(self.weights, self.theta1)

        relearnIdx = np.where((q2 > np.array(self.weights)) * (np.array(self.weights) >= q1))[0]
        relearnIdx = random.sample(list(relearnIdx), int((1-self.beta)*len(relearnIdx)))
        for i in relearnIdx:
            self.relearnLearner(i)

        dropIdx = np.where((q1 > np.array(self.weights)))[0]
        for i in dropIdx:
            self.dropLearner(i)

    def partial_fit(self, X, X_mask, Y):

        self.instance_buffer.add(X, X_mask, Y)
        self.updateFeatureSet(X_mask)
                
        y_pred, y_logits = self.predict(X)
        self.window.add(int(y_pred == Y))
        if self.window.errorRate() > self.gamma:
            self.initNewLearner()
            self.updateGamma()

        self.update(X, Y)

        return y_pred, y_logits