import numpy as np
import math
import random
from tdigest import TDigest

def calcHB(delta, windowSize):
    return math.sqrt(math.log(1/delta) / (2 * float(windowSize)))

class Queue:
    def __init__(self, n):
        self.n = n
        self.queue = []
    
    def add(self, val):
        self.queue.append(val)
        if(len(self.queue) > self.n):
            self.queue.pop(0)
    
    def mean(self):
        m  = 1.0 * sum(self.queue)/len(self.queue)
        return m

class FeatStats:
    def __init__(self, n, x=None, y=None, numClasses = 2):
        self.slidingWindow = Queue(n)
        self.seenInstances = 0
        self.digestPerClass = {i:TDigest() for i in range(numClasses)}
        self.minValue = None
        self.maxValue = None
        self.classCounts = {i:0.0 for i in range(numClasses)}

        if( (x is not None) and (y is not None) ):
            self.update(x, y)

    def update(self, x=None, y=None):
        if( (x is not None) and (y is not None) ):
            y = int(y)
            self.slidingWindow.add(True)
            self.seenInstances += 1
            self.digestPerClass[y].update(x)
            
            if (self.minValue is None): self.minValue = x
            else: self.minValue = min(self.minValue, x)
            
            if (self.maxValue is None): self.maxValue = x
            else: self.maxValue = max(self.maxValue, x)

            self.classCounts[y]+=1
        
        else:
            self.slidingWindow.add(False)

class Stump:
    def __init__(self, minValue, maxValue, digestPerClass, classCounts):
        self.age=0
        self.threshold = minValue + random.random() * (maxValue - minValue)
        self.gini, self.classDistAbove, self.classDistBelow = self.calcApproxGini(digestPerClass, classCounts)

    def calcApproxGini(self, digestPerClass, classCounts):

        totalCount = 0.0
        for count in classCounts:
            totalCount += count
        
        
        classDistBelow = dict()
        classDistAbove = dict()
        for Class, digest in digestPerClass.items():
            classDistBelow[Class] = digest.cdf(self.threshold)  #P(fi<Xj | C)
            classDistAbove[Class] = 1 - classDistBelow[Class]   #P(fi>Xj | C)
        
        countBelow, countAbove = 0.0, 0.0
        for Class, digest in digestPerClass.items():
            countBelow += classDistBelow[Class] * classCounts[Class] #NB
            countAbove += classDistAbove[Class] * classCounts[Class] #NA
        
        normalizedClassDistBelow, normalizedClassDistAbove = dict(), dict()
        epsilon = np.finfo(float).eps
        for Class in classCounts:
            normalizedClassDistBelow[Class] = classDistBelow[Class] * classCounts[Class] / (countBelow+epsilon) # P(fi<Xj | C) * (Nc) / NB
            normalizedClassDistAbove[Class] = classDistAbove[Class] * classCounts[Class] / (countAbove+epsilon) # P(fi>Xj | C) * (Nc) / NA
        
        giniBelow, giniAbove = 0.0, 0.0
        for Class in classCounts:
            giniBelow += normalizedClassDistBelow[Class] ** 2
            giniAbove += normalizedClassDistAbove[Class] ** 2
        
        giniGain = 0.0
        if (not np.isnan(giniBelow)) and (not np.isnan(giniAbove)):
            giniGain = (giniAbove * countAbove / totalCount) + (giniBelow * countBelow / totalCount)
        
        return giniGain, normalizedClassDistBelow, normalizedClassDistAbove
    
    def predict(self, x):
        self.age += 1
        if x < self.threshold:
            return self.classDistBelow
        else:
            return self.classDistAbove

class FeatureForest:
    def __init__(self, n, feature, featStats, numClasses):
        self.feature = feature
        self.numClasses = numClasses
        self.decisionStumps = [Stump(minValue      =featStats.minValue,
                                      maxValue      =featStats.maxValue,
                                      digestPerClass=featStats.digestPerClass,
                                      classCounts   =featStats.classCounts)] * n

    def predict(self, x):
        distAggregate = np.array([0.0]*self.numClasses)
        for stump in self.decisionStumps:
            predDist = stump.predict(x)
            predDist = {key:predDist[key] for key in sorted(predDist)}
            predDist = np.array(list(predDist.values()))
            distAggregate += predDist
        
        return np.argmax(distAggregate), max(distAggregate)


class ORF3V:
    def __init__(self, forestSize=200, replacementInterval=50, replacementChance=0.3, 
                windowSize = 200, updateStrategy = "oldest", alpha = 0.1, delta = 0.001, 
                Xs: np.array=None, X_masks: np.array=None, Ys: np.array=None, numClasses = 2):

        self.forestSize = forestSize
        self.replacementInterval = replacementInterval
        self.replacementChance = replacementChance
        self.windowSize = windowSize
        self.updateStrategy = updateStrategy
        self.alpha = alpha
        self.delta = delta
        self.numClasses = numClasses

        self.counter = 0
        self.featStats = dict()
        self.firstOccurance = dict()
        self.currentFeatures = list()
        self.featureSet = set()
        self.featureForest = dict()
        self.weights = dict()

        self.hb = calcHB(self.delta, self.windowSize)

        self.initialUpdate(Xs, X_masks, Ys)

    def initialUpdate(self, Xs, X_masks, Ys):
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)
            X_masks = X_masks.reshape(1, -1)
            Ys = Ys.reshape(1, -1)
        for x, x_mask, y in zip(Xs, X_masks, Ys):
            self.update(x, x_mask, y)

    def predict(self, X, X_mask):
        self.currentFeatures = np.where(X_mask)[0]
        classProbs = np.array([0.0] * self.numClasses)
        for feature in self.currentFeatures:
            if feature in self.featureSet:
                ff = self.featureForest[feature]
                weight = self.weights[feature]

                classProbs += weight * ff.predict(X[feature])
        
        return np.argmax(classProbs), max(classProbs)

    def update(self, X, X_mask, Y):

        '''
        1. Updating the feature statistics,
        2. Pruning of feature forests
        3. Generating feature forests
        4. Updating weights
        5. Replacing decision stumps
        '''

        self.counter += 1
        # Features in Current instance
        self.currentFeatures = set(np.where(X_mask)[0])
        # Feature present in current instance but not in FeatureSet
        newFeatures = self.currentFeatures.difference(self.featureSet)
        # Update featureSet
        self.featureSet.update(newFeatures)

        # Update FeatureStats for existing features
        self.updateFeatureStats(X, Y)

        # Prune FeatureForest for vanished features
        self.pruneForests()

        # Generate new FeatureForest for new features
        self.generateForest(newFeatures, X, Y)

        # Update weights for each feature
        self.updateWeights(X, Y)

        # Replace stupmps from forest according to replacementStrategy
        if self.counter > 10 and self.counter%self.replacementInterval == 0:
            self.replaceStumps()

    def updateFeatureStats(self, X, Y):
        for feature in self.featStats:
            # Feature persent in FeatureSet and current instance
            if feature in self.currentFeatures:
                self.featStats[feature].update(X[feature], Y)
            
            # Feature persent in FeatureSet but not in current instance
            else:
                self.featStats[feature].update()
        return
    
    def pruneForests(self):
        if (self.counter > self.windowSize) and (self.counter%self.replacementInterval == 0):
            for feature, featStats in self.featStats.items():
                
                if featStats.seenInstances > self.windowSize:

                    totalMean = (featStats.seenInstances)/(self.counter-self.firstOccurance[feature])
                    windowMean = featStats.slidingWindow.mean()
                    if totalMean-windowMean > self.hb:
                        self.prune(feature)
        return

    def prune(self, feature):
        del self.featureForest[feature], self.weights[feature], self.firstOccurance[feature], self.featureSet[feature]
        return  

    def generateForest(self, newFeatures, X, Y):

        for feature in newFeatures:
                self.firstOccurance[feature] = self.counter
                self.weights[feature] = 1.0
                self.featStats[feature] = FeatStats(self.windowSize, X[feature], Y, self.numClasses)
                self.featureForest[feature] = FeatureForest(self.forestSize, feature, self.featStats[feature], self.numClasses)
        
        return
    
    def updateWeights(self, X, Y):
        for feature in self.currentFeatures:
            if feature in self.featureForest:
                predClass, _ = self.featureForest[feature].predict(X[feature])
                self.weights[feature] = (2*self.alpha*(predClass == Y) + self.weights[feature])/(1+self.alpha)
        return

    def replaceStumps(self):
        if self.updateStrategy == "random":
            self.replaceRandomStumps()
        else:
            self.replaceOldestStump()
    
    def replaceRandomStumps(self):
        for feature, ff in self.featureForest.items():
            for i in range(len(ff.decisionStumps)):
                if random.random() < self.replacementChance:
                    ff.decisionStumps[i] = Stump(minValue       =self.featStats[feature].minValue,
                                                  maxValue      =self.featStats[feature].maxValue,
                                                  digestPerClass=self.featStats[feature].digestPerClass,
                                                  classCounts   =self.featStats[feature].classCounts)
        return
    
    def replaceOldestStump(self):
        for feature, ff in self.featureForest.items():
            oldestAge, oldestIdx = 0, -1
            for i, stump in enumerate(ff.decisionStumps):
                if(stump.age > oldestAge):
                    oldestAge = stump.age
                    oldestIdx = i
            
            ff.decisionStumps[oldestIdx] = Stump(minValue      =self.featStats[feature].minValue,
                                                 maxValue      =self.featStats[feature].maxValue,
                                                 digestPerClass=self.featStats[feature].digestPerClass,
                                                 classCounts   =self.featStats[feature].classCounts)

    def partial_fit(self, X, X_mask, Y):

        Y_pred, Y_logit = self.predict(X, X_mask)

        self.counter += 1
        self.update(X, X_mask, Y)

        return Y_pred, Y_logit