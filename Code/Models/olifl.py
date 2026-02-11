# Code from https://github.com/Rohit102497/packetLSTM_HaphazardInputs/blob/main/Baselines/Models/OLIFL.py
# Code from https://github.com/youdianlong/OLIFL

import numpy as np
import random,time
import copy,random
from Utils.utils import seed_everything
from sklearn import preprocessing
from tqdm.auto import tqdm
import math

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # Depricated
import warnings
warnings.filterwarnings('error', category=np.exceptions.VisibleDeprecationWarning)       

def findCommonKeys(classifier, row):  # find the common keys of two dictionaries
    return (set(classifier.keys()) & set(row.keys()))


def findDifferentKeys(dict1, dict2):
    return (set(dict1.keys()) - set(dict2.keys()))


def subsetDictionary(dictionary, intersection):  # extract subset of key-val pairs if in
    return dict((value, dictionary[value]) for value in intersection)


def dict2NumpyArray(dictionary):
    return np.array([dictionary[key] for key in sorted(dictionary.keys())])


def dotDict(dict1, dict2):
    returnValue = 0
    for key in dict1:
        returnValue += dict1[key] * dict2[key]
    return returnValue


def NumpyArray2Dict(numpyArray, keys):
    return {k: v for k, v in zip(keys, numpyArray)}

class OLIFL:
    def __init__(self, C, option,seed):
        self.C = C
        self.option = option
        self.seed = seed
        seed_everything(self.seed)

        self.rounds = 1

    def custom_set_data(self,X, Y, X_haphazard, mask):
        self.X = []
        for t in range(len(Y)):
            dct = {}
            for f in range(X.shape[1]):
                if mask[t][f]:
                    dct[f]=X_haphazard[t][f]
            self.X.append(dct)
        self.y = Y
    def parameter_set(self, X,loss):
        inner_product=dotDict(X, X)
        if inner_product == 0:
            inner_product = 1

        if self.option == 0: return loss / inner_product
        if self.option == 1: return np.minimum(self.C, 2*loss / inner_product)

    def set_classifier(self):
        self.weights = {key: 0 for key in self.X[0].keys()}
        self.u_count={key: 1 for key in self.X[0].keys()}
        self.stability = self.X[0]
        self.count = dict()
        self.A_ =  dict()
        self.A  =  dict()
        self.keyCount=dict()
        self.n_num = 0
        self.e_num = 0
        self.s_num = 0
        self.sum_loss=0
    def update_stability(self,X):
        e_stability=self.stability
        for key in X.keys():
            if key not in self.count.keys():
                self.count[key] = 1
                self.keyCount[key]=1
                self.A_[key]=X[key]
                self.A[key]=X[key]
                self.stability[key]=0.000001
            else:

                self.count[key] +=1
                self.A_[key]=self.A[key]
                self.A[key]=self.A[key] + (X[key] - self.A[key]) / self.count[key]
                self.stability[key]=(self.count[key]-1)/self.count[key]**2*(X[key]-self.A_[key])**2+(self.count[key]-1)/self.count[key]*self.stability[key]
        return e_stability
    def upKeyCount(self,X): 
        sum1 = 0
        e_KeyCount=self.keyCount
        for key in X.keys():
            sum1 += self.stability[key]
        for key in X.keys():
            self.keyCount[key] = self.stability[key] / sum1
        return e_KeyCount

    def predict(self,X):
        y_pre=np.sum(np.array([X[k] * self.keyCount[k] * self.weights[k] for k in X.keys()],dtype=object))
        return y_pre
    



    def expand_space(self,X):
        self.n_keys=0
        self.e_keys=0
        self.s_keys=0
        e_weights=self.weights
        for key in findDifferentKeys(X, self.weights):
            self.weights[key] = 0
            self.u_count[key] = 1
            self.n_keys+=1
        for key in findDifferentKeys(self.weights,X):
            X[key] = 0
            self.e_keys+=1
        for key in findCommonKeys(X,self.weights):
            self.u_count[key] += 1
            self.s_keys=+1
        return X,e_weights

    def fit(self,X, Y, X_haphazard, mask):#
        self.custom_set_data(X, Y, X_haphazard, mask)
        for i in range(self.rounds):
            start = time.time()
            Y_pred=[] 
            Y_logits=[]
            
            train_error, train_loss = 0, 0
            train_error_vector, train_loss_vector, train_acc_vector = [], [], []

            TN = 0
            self.set_classifier()

            l=0
            ###########Change dataformat to required dictionary format#############
            for t in tqdm(range(0, len(self.y))):
                l+=1
                row, _ =self.expand_space(self.X[t])
                if len(row) == 0:
                    train_error_vector.append(train_error / (l + 1))
                    train_loss_vector.append(train_loss / (l + 1))
                    train_acc_vector.append(1 - train_error / (l + 1))
                    continue
                sta_ = self.update_stability(row)
                KC_ = self.upKeyCount(row)
                y_value = self.predict(row)
                y_hat = np.sign(y_value)

                y_hat_logit = 1/(1+np.exp(y_hat))
                Y_logits.append(y_hat_logit)
                y_up = 2*self.y[t]-1

                if y_hat != y_up:
                    train_error += 1
                    if self.y[i] == 1:
                        TN += 1
                loss = (np.maximum(0, (1 - (y_up) * y_value)))
                tao = self.parameter_set(row, loss)
                self.weights=self.upweight(tao,row,y_up)
                train_error_vector.append(train_error / (l + 1))
                train_loss += loss
                train_loss_vector.append(train_loss / (l + 1))
                train_acc_vector.append(1 - (train_error / (l + 1)))
                Y_pred.append([0 if y_hat==-1.0 else 1])
            x = type(Y_logits[0])
            for y in range(len(Y_logits)):
                if x!=type(Y_logits[y]):
                    Y_logits[y] = Y_logits[y][0]
            return Y_pred, Y_logits

    def upweight(self,tao,X,y):


        return {key: self.weights[key] + tao * (y) *X[key]*self.keyCount[key] for key in X.keys()}