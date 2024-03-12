# This is the implementation for a binary classification problem with
# classes denoted as {1, 0} for positive and negative classes respectively

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
    # calculates the chi square statistics of a feature
    numerator = N * (A*D - C*B)**2
    denominator = (A+C) * (B+D) * (A+B) * (C+D)
    if numerator == 0:
        return 0
    else:
        return numerator/denominator


class NB3:
    def __init__(self, Documents=None, Classes=None):

        self.Vocabulary = set()     # All words seen by the model till time t
        self.WordStats = dict()     # key: word (feature)
                                    # value: 2x2 matrix # (0, 0) -> number of times word is absent in class 0 document
                                                        # (0, 1) -> number of times word is absent in class 1 document
                                                        # (1, 0) -> number of times word is present in class 0 document
                                                        # (1, 1) -> number of times word is present in class 1 document
        self.Features = dict()      # key : word
                                    # value : evaluation score
        self.FeatureList = list()   # list of words (features) sorted according to evaluation score

        self.classifier = Classifier()
        self.Nc = np.array([0.0, 0.0]) # Number of documents corresponding to each class

        if (Documents is not None) and (Classes is not None):
            self.InitialTraining(Documents, Classes)
        else:
            self.Nc += 1
            # Incase of no pretraining, calculation of probability of each word faces 'Divide by Zero' giving out NaN values,
            # (line 127) in the Update function of class Classifier
            # to handle this, we initialized Nc by [1, 1] instead of [0, 0]

    # Algorithm InitialTraining
    def InitialTraining(self, Documents, Classes):

        for Document, DocClass in zip(Documents, Classes):
            self.Update(Document, DocClass)
    
    # Function to update WordStats
    def UpdateWordStats(self, Document, DocClass):
        
        for Word in self.Vocabulary:
            if Word in Document:
                self.WordStats[Word][1][DocClass] += 1.0
            else:
                self.WordStats[Word][0][DocClass] += 1.0

    # Algorithm Update
    def Update(self, Document, DocClass):
        DocClass = int(np.squeeze(DocClass))
        # Update Number of documents in each class
        self.Nc[DocClass] += 1.0

        # Update Vocabulary and Initiate WordStats for new words.
        for Word in set(Document):
            if Word not in self.Vocabulary:
                self.Vocabulary.add(Word)
                self.WordStats[Word] = np.zeros((2, 2))
        
        # Update WordStats
        self.UpdateWordStats(Document, DocClass)
        
        # Evaluate and calculate X2 statistic of a word
        for Word in self.Vocabulary:
            Evaluation = EvaluateFeature(Word, self.WordStats, sum(self.Nc))
            self.Features[Word] = Evaluation
        
        # Sort features in decreasing order of their evaluation value
        self.FeatureList = [Word for Word, Evaluation in sorted(self.Features.items(), key=lambda x:x[1], reverse=True)]
        
        # Build Classifier
        self.classifier.Update(self.Vocabulary, self.WordStats, self.Nc)

    # Make Predictions
    def predict(self, Document, NumToSelect):
        return self.classifier.use(Document, self.FeatureList[:NumToSelect])
    
    # Partial Fit function
    def partial_fit(self, Document, DocClass, NumToSelect):

        # Make Prediction
        result = self.predict(Document, NumToSelect)

        # Update Classifier
        self.Update(Document, DocClass)

        return result


class Classifier:
    def __init__(self):
        
        self.Pc = np.array([0.5, 0.5])  # Probability of occurance of each class
        self.P = dict()                 # Dictionary to store probability of each word (feature) w.r.t each class
                                        # key : word | value : [prob. wrt class 0, prob. wrt class 1] (2d vector)

    def Update(self, Vocabulary, WordStats, Nc):

        # Probability of Document belonging to class nc
        self.Pc = np.array(Nc)/sum(Nc)

        # Conditional probability of each word
        for Word in Vocabulary:
            self.P[Word] = WordStats[Word][1] / (Nc)
            # In case of no pretraining, to avoid divide by zero situation, producing NaN outputs,
            # Nc is initalized with [1, 1] instead of [0, 0].
    
    def score(self, Document, FeatureList):
        scores = self.Pc
        # print("Scores: ", scores)
        if len(self.P) != 0:
            for Word in Document:
                if Word in FeatureList:
                    if Document[Word] != 0: # The value of a continous features can be 0
                        # print("Value: ", Document[Word])
                        scores *= self.P[Word]*abs(Document[Word]) # taking abs to convert negative feature value to positive
        return scores
    
    def use(self, Document, FeatureList):
        scores = self.score(Document, FeatureList)
        # print("Scores: ", scores)
        if sum(scores) == 0:
            scores += 0.5
        scores = scores/sum(scores)
        pred_class =  np.argmax(scores)
        return pred_class, scores