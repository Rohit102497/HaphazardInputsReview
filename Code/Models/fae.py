import numpy as np

def EvaluateFeature(Word, WordStats, N):
    # A : number of times w and c co-occur
    # B : number of times w occurs without c
    # C : number of times c occurs without w
    # D : number of times neither c nor w occur
    # N : total number of documents

    # Returns an evaluation score

    positive_class, negative_class = 1, 0

    A = WordStats[Word][1][positive_class]
    B = WordStats[Word][1][negative_class]
    C = WordStats[Word][0][positive_class]
    D = WordStats[Word][0][negative_class]

    Evaluation = X2(A, B, C, D, N)
    return Evaluation

def X2(A, B, C, D, N):
    #Calculates the Chi Square statistics of a feature
    numerator = N * (A*D - C*B)**2
    denominator = (A+C) * (B+D) * (A+B) * (C+D)
    if numerator == 0:
        return 0
    else:
        return numerator/denominator


class ACC_Q:
    '''A queue maintaining the accuracy over last n and 2n instances'''
    def __init__(self, N=50):
        self.N = N

        self.accuracies = []
        self.Acc_over_N = 0
        self.Acc_over_2N = 0
    
    def Push(self, val):
        self.accuracies.append(val)
        if len(self.accuracies) > 2*self.N:
            self.accuracies.pop(0)
        
        self.Acc_over_N = sum(self.accuracies[-self.N:])
        self.Acc_over_2N = sum(self.accuracies[-2 * self.N:])


class LEARNER:
    def __init__(self, Document=None, DocClass=None, Features=[], N=50, data_name = None):
        
        self.Nc = np.array([0.0, 0.0]) # Number of documents corresponding to each class
        
        self.Nc += 1
        # Calculation of probability of each word faces 'Divide by Zero' condition giving out NaN values,
        # (line 114) in the UpdateProbabilities function of this class
        # to handle this, we initialized Nc by [1, 1] instead of [0, 0]

        self.Vocabulary = Features # All words seen by the model till time t
        self.WordStats = {Word : np.zeros((2, 2)) for Word in self.Vocabulary} # value: 2x2 matrix  # (0, 0) -> number of times word is absent in class 0 document
                                                                                                    # (0, 1) -> number of times word is absent in class 1 document
                                                                                                    # (1, 0) -> number of times word is present in class 0 document
                                                                                                    # (1, 1) -> number of times word is present in class 1 document
        
        self.Pc = np.array([0.5, 0.5]) # Probability of occurance of each class
        self.P = dict() # Dictionary to store probability of each word (feature) w.r.t each class
                        # key : word | value : [prob. wrt class 0, prob. wrt class 1] (2d vector)

        self.acc = ACC_Q(N) # Queue of accuracy over last n instances
        self.Accuracy = 0
        self.age = 0
        self.probation = 0
        self.data_name = data_name

        if (Document is not None and DocClass is not None):
            self.Accuracy = 1 # If pre-trained, accuracy considered to be 100% till that instant
            self.Update(Document, DocClass)

    def score(self, Document):
        scores = self.Pc
        if len(self.P) != 0:
            for Word in Document:
                if Word in self.Vocabulary:
                    div = 1.0
                    if self.data_name in ['crowdsense_c3', 'crowdsense_c5']:
                        div = 1000.0
                    # For crowdsense data the scores shoots up to infinity. To curtail that we divide the input values by 1000.
                    scores *= self.P[Word]*(Document[Word]/div)

        return scores
    
    # Performs prediction of Learner and updates accuracy
    def use(self, Document, DocClass = None):
        scores = self.score(Document)
        pred_class =  np.argmax(scores)
        logit = scores[pred_class]

        if DocClass is not None:
            self.acc.Push(int(pred_class == DocClass))
            self.Accuracy = self.acc.Acc_over_N
            
        return pred_class, logit
    
    def UpdateWordStats(self, Document, DocClass):
        # Update the WordStat of every word(feature) in vocabulary of the learner
        for Word in self.Vocabulary:
            if Word in Document:
                self.WordStats[Word][1][DocClass] += 1
            else:
                self.WordStats[Word][0][DocClass] += 1
    
    def UpdateProbabilities(self):
        # Probability of Document belonging to class nc
        self.Pc = np.array(self.Nc)/sum(self.Nc)

        # Conditional probability of each word
        for Word in self.Vocabulary:
            self.P[Word] = self.WordStats[Word][1] / (self.Nc)
    
    def Update(self, Document, DocClass):
        # Wrapper to perform updates and call other update functions
        self.age += 1

        self.Nc[DocClass] += 1
        self.UpdateWordStats(Document, DocClass)
        self.UpdateProbabilities()    


class FAE:

    def __init__(self, m=5, p=3, f=0.15, r=10, N=50, M=250, data_name = None, Document=None, DocClass=None):
        
        # Initialize parameters
        self.m = m # (maturity) Number of instances needed before a learner’s classifications are used by the ensemble
        self.p = p # (probation time) is the number of times in a row a learner is allowed to be under the threshold before being removed
        self.f = f # (feature change threshold) is the threshold placed on the amount of change between the youngest learner’s set of features (yfs) and the top M features (mfs);
        self.r = r # (growth rate) is the number of instances between when the last learner was added and when the ensemble’s accuracy is checked for the addition of a new learner
        self.N = N # Number of instances over which to compute an accuracy measure;
        self.M = M # Number of features selected by the feature selection algorithm for a newly created learner

        self.FeatureDict = dict()   # key : word
                                    # value : evaluation score
        
        self.FeatureList = list() # list of words (features) sorted according to evaluation score
        
        self.TopMFeatures = list() # list of top M words (features) with highest evaluation score
        
        self.Vocabulary = set() # All words seen by the model till time t
        
        self.WordStats = dict() # key: word (feature)
                                # value: 2x2 matrix # (0, 0) -> number of times word is absent in class 0 document
                                                    # (0, 1) -> number of times word is absent in class 1 document
                                                    # (1, 0) -> number of times word is present in class 0 document
                                                    # (1, 1) -> number of times word is present in class 1 document
        
        self.Nc = [0.0, 0.0] # Number of documents corresponding to each class

        # Initialize other required attributed
        self.acc = ACC_Q(self.N)
        self.probation = np.array([])
        self.age = np.array([])
        self.data_name = data_name

        # Initialize the first learner
        if(Document is not None and DocClass is not None):
            self.InitialTraining(Document, DocClass)
            self.learners = [LEARNER(Document=Document, DocClass=DocClass, Features=self.TopMFeatures,
                                      N=self.N, data_name = self.data_name)]
        else:
            self.learners = [LEARNER(Features=self.TopMFeatures, N=self.N, data_name = self.data_name)]
        
        self.threshold = self.get_threshold()
    
    def UpdateWordStats(self, Document, DocClass):
        # Update the WordStat of every word(feature) in vocabulary of the learner
        for Word in self.Vocabulary:
            if Word in Document:
                self.WordStats[Word][1][DocClass] += 1
            else:
                self.WordStats[Word][0][DocClass] += 1
    
    def InitialTraining(self, Document, DocClass):
        self.Nc[DocClass] += 1

        # Create Initial Vocabulary
        self.Vocabulary = set(Document)

        # Create Initial WordStats and update
        self.WordStats = {Word: np.zeros((2, 2)) for Word in self.Vocabulary}
        self.UpdateWordStats(Document, DocClass)

        # Evaluate and calculate X2 statistic of a word
        for Word in self.Vocabulary:
            Evaluation = EvaluateFeature(Word, self.WordStats, sum(self.Nc))
            self.FeatureDict[Word] = Evaluation
        
        # Sort features in decreasing order of their evaluation value
        self.FeatureList = [Word for Word, Evaluation in sorted(self.FeatureDict.items(), key=lambda x:x[1], reverse=True)]        
        self.TopMFeatures = self.FeatureList[:self.M]

    def get_threshold(self):
        learner_accuracies = [learner.Accuracy for learner in self.learners]
        th = (max(learner_accuracies) + min(learner_accuracies))/2.0
        return th
    
    def predict(self, Document, DocClass = None):
        
        age = np.array([learner.age for learner in self.learners])
        probation = np.array([learner.probation for learner in self.learners])

        mask = ((age >= self.m) * (probation == 0)).astype(int)
        if sum(mask) == 0:
            weights = age
        else:
            weights = np.array([learner.Accuracy for learner in self.learners])

        scores = np.array([0.0, 0.0])
        preds = np.array([0.0, 0.0])

        for weight, learner in zip(weights, self.learners):
            pred, logit = learner.use(Document, DocClass)

            preds[pred] += weight*pred
            scores[pred] += weight*logit
        
        y_pred = np.argmax(preds)
        y_logit = scores[y_pred]

        return y_pred, y_logit
   
    def Update(self, Document, DocClass):
        # Update Threshold
        self.threshold = self.get_threshold()
        
        # Update Probations
        for learner in self.learners:
            if learner.Accuracy < self.threshold:
                learner.probation += 1
            else:
                learner.probation = 0
        
        # Remove learners whose probation is more than p
        for i, learner in enumerate(self.learners):
            if learner.probation >= self.p:
                self.learners.pop(i)
        
        # Create new learner on current instance
        self.learners.append(LEARNER(Document=Document, DocClass=DocClass, Features=self.TopMFeatures, data_name=self.data_name))

    def UpdateFeatureSet(self, Document):
        # Update Vocabulary and Initiate WordStats for new words.
        for Word in set(Document):
            if Word not in self.Vocabulary:
                self.Vocabulary.add(Word)
                self.WordStats[Word] = np.zeros((2, 2))
        
        # Evaluate and calculate X2 statistic of a word
        for Word in self.Vocabulary:
            Evaluation = EvaluateFeature(Word, self.WordStats, sum(self.Nc))
            self.FeatureDict[Word] = Evaluation
        
        # Sort features in decreasing order of their evaluation value
        self.FeatureList = [Word for Word, Evaluation in sorted(self.FeatureDict.items(), key=lambda x:x[1], reverse=True)]        
        self.TopMFeatures = self.FeatureList[:self.M]

    def partial_fit(self, Document, DocClass):
        DocClass = int(DocClass)

        # Update Ensemble Accuracy
        y_pred, y_logit = self.predict(Document, DocClass)
        ensemble_accuracy = int(y_pred == DocClass)
        self.acc.Push(ensemble_accuracy)

        # Update Feature Set
        self.UpdateFeatureSet(Document)

        # Train all learners on current instance
        for learner in self.learners:
            learner.Update(Document, DocClass)
        
        # Calculate change in features
        youngest_learner = self.learners[-1]
        
        nfs = set(self.TopMFeatures)
        yfs = set(youngest_learner.Vocabulary)

        delta = len(nfs.symmetric_difference(yfs)) / self.M
        
        if (delta>self.f) or ( (youngest_learner.age>self.r) and (self.acc.Acc_over_N < self.acc.Acc_over_2N) ):
           
            self.Update(Document, DocClass)
        
        return y_pred, y_logit
    
    def learner_count(self):
        return len(self.learners)