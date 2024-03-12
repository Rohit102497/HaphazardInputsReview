import random
import numpy as np

def remove_elements_by_indices(input_list, indices_to_remove):
    # Create a new list with elements at indices not in indices_to_remove
    result_list = [element for index, element in enumerate(input_list) if index not in indices_to_remove]
    return result_list

# Helper function for custom implementation of DescisionStump
def cal_partial(l_y, l, y_sign):
    mask= l_y == 0
    flag = np.array([0]*l)
    flag[~mask] = y_sign[~mask]/l_y[~mask]
    flag[mask] = 0
    return flag

# Implementation of fixed size queue
# Automatically pops elements when new element inserted in a full queue
class Queue:
    def __init__(self, n: int):
        self.queue = []
        self.n = n
    
    def push(self, instance: any):
        self.queue.append(instance)
        if len(self.queue) > self.n:
            self.queue.pop(0)

# Buffer to store instances and retrive them featurewise
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


# Custom implementaion of DescisionStump to suit an online learning setting
class DecisionStump:
    def __init__(self):
        self.best_gini = float('inf')
        self.accepted_features = set()
        self.feature_index = None
        self.threshold = None
        self.prediction_positive = None
        self.prediction_negative = None
        self.probability_positive = None
        self.probability_negative = None
    
    def fit(self, X: np.array, y: np.array, feature_index: int):
        y = y.reshape(-1)
        self.accepted_features.add(feature_index)

        total_samples = len(y)
        if total_samples != 0:
            X = X.reshape(total_samples, 1)
            y = np.array([y]*total_samples)
            mask = X.T <= X
            l_y_positive = np.sum(mask, axis = 1)
            l_y_negative = total_samples - l_y_positive
            y_pos_1 = np.sum(y*mask, axis = 1)
            y_pos_0 = l_y_positive - y_pos_1
            y_neg_1 = np.sum(y*(~mask), axis = 1)
            y_neg_0 = l_y_negative - y_neg_1

            y_pos_1_partial = cal_partial(l_y_positive, total_samples, y_pos_1)
            y_pos_0_partial = cal_partial(l_y_positive, total_samples, y_pos_0)
            y_neg_1_partial = cal_partial(l_y_negative, total_samples, y_neg_1)
            y_neg_0_partial = cal_partial(l_y_negative, total_samples, y_neg_0)

            gini_positive = 1 - ((y_pos_1_partial)**2 + (y_pos_0_partial)**2)
            gini_negative = 1 - ((y_neg_1_partial)**2 + (y_neg_0_partial)**2)
            gini = (l_y_positive/ total_samples)*gini_positive + (l_y_negative/ total_samples)*gini_negative

            best_idx = np.argmin(gini)

            self.best_gini = gini[best_idx]
            self.feature_index = feature_index
            self.threshold = X[best_idx]

            mask = mask[best_idx].reshape(-1)
            y = y[0].reshape(-1)
            y_positive = y[mask]
            y_negative = y[~mask]

            self.prediction_positive = self.get_majority_class(y_positive)
            self.prediction_negative = self.get_majority_class(y_negative)
            self.probability_positive = sum(y_positive == self.prediction_positive)/len(y_positive) if len(y_positive)!=0 else 0.0
            self.probability_negative = sum(y_negative == self.prediction_negative)/len(y_negative) if len(y_negative)!=0 else 0.0

    def predict(self, X: np.array) -> int:
        return (self.prediction_positive, self.probability_positive) if X[self.feature_index] <= self.threshold \
                else (self.prediction_negative, self.probability_negative)

    def get_majority_class(self, y: list) -> int:
        unique_labels = set(y)
        if not unique_labels:
            return 0
        return int(max(unique_labels, key=y.tolist().count))

    def splitDescision(self) -> int:
        return self.feature_index

# Window for each stump to keep track of accuracy
class Window(Queue):
    def __init__(self, n):
        super().__init__(n)
    
    def add(self, accuracy):
        self.push(accuracy)
    
    def errorRate(self):
        return 1 - (sum(self.queue)/len(self.queue))

class DynFo:
    def __init__(self, Xs: np.array=None, X_masks: np.array=None, Ys: np.array=None,
                 alpha = 0.5, beta = 0.3, delta = 0.01, epsilon = 0.001, 
                 gamma = 0.7, M = 1000, N = 1000, theta1=0.05, theta2=0.6, num_classes = 2):
        
        # alpha - Impact on weight update
        # beta - Probability to keep the weak learner in ensemble
        # delta - Fraction of features to consider (bagging parameter)
        # epsilon - Penalty if the split of decision stump is not in the current instance
        # gamma - Threshold for error rate
        # M - Number of learners in the ensemble
        # N - Buffer size of instances
        # theta1 - Lower bounds for the update strategy
        # theta2 - Upper bounds for the update strategy
        # num_classes - Number of target classes
        
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

        self.weights = list()
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
        newLearner = DecisionStump()
        for feature in accepted_features:
            X, Y = self.instance_buffer.get(feature)
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
        learner = self.learners[i]
        for feature in accepted_features:
            X, Y = self.instance_buffer.get(feature)
            learner.fit(X, Y, feature)

    def dropLearner(self, i):
        self.learners.pop(i)
        self.weights.pop(i)
        self.acceptedFeatures.pop(i)
        # Sanity Check:
        assert (len(self.weights) == len(self.learners)) and (len(self.acceptedFeatures) == len(self.learners))
    
    def predict(self, X):
        # Take the weighted sum of every learner (decision stump)
        wc = np.array([0.0] * self.num_classes)
        for learner, weight in zip(self.learners, self.weights):
            if learner.splitDescision() in self.currentFeatures:
                pred, prob = learner.predict(X)
                wc[pred] += (weight * prob)
                opp_pred = 1 if pred==0 else 0
                wc[opp_pred] += (weight*(1-prob))
        
        return np.argmax(wc), max(wc)

    def update(self, X, Y):
        for j in range(len(self.weights)):
            
            # Peanalize learner if split decision not in their featureset, otherwise update learner weight
            if self.learners[j].splitDescision() not in self.currentFeatures:
                self.weights[j] -= self.epsilon
            else:
                i = int(np.squeeze(self.learners[j].predict(X)[0] == Y))
                self.weights[j] = (i*2*self.alpha + self.weights[j]) / (1 + self.alpha)
        
        # get index of learners to relearn
        q2 = np.quantile(self.weights, self.theta2)        
        relearnIdx = np.where((self.theta1 <= np.array(self.weights)) * (np.array(self.weights) < q2))[0]
        relearnIdx = random.sample(list(relearnIdx), int((1-self.beta)*len(relearnIdx)))
        for i in relearnIdx:
            self.relearnLearner(i)
            self.weights[i] = 1

        # get index of learners to drop
        dropIdx = np.where((np.array(self.weights) < self.theta1))[0]
        for i in range(len(dropIdx)):
            self.dropLearner(dropIdx[i] - i)
    
    def partial_fit(self, X, X_mask, Y):

        self.instance_buffer.add(X, X_mask, Y)
        self.updateFeatureSet(X_mask)
                
        y_pred, y_logits = self.predict(X)
        self.window.add(int(np.squeeze(y_pred == Y)))
        if self.window.errorRate() > self.gamma:
            self.initNewLearner()
            self.updateGamma()

        self.update(X, Y)

        return y_pred, y_logits