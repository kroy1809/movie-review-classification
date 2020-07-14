import sys

import numpy as np
from Eval import Eval
import collections

from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA=ALPHA
        #TODO: Initalize parameters
        self.feat_pos_count = []
        self.feat_neg_count = []
        self.prior = {}
        self.Train(X,Y)

    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        #Feature vector creation
        feat_pos = X[(Y>0),:].toarray()
        self.feat_pos_count = feat_pos.sum(axis = 0)
        
        feat_neg = X[(Y<0),:].toarray()
        self.feat_neg_count = feat_neg.sum(axis = 0)
        
        #Prior calculation
        prior = collections.Counter(Y)
        self.prior = {k: v / total for total in (sum(prior.values()),) for k, v in prior.items()}
        
    def Predict(self, X):
        #TODO: Implement Naive Bayes Classification
        X_array = X.toarray()
        file_nos = X_array.shape[0]
        pred_Y = np.zeros(file_nos)

        #Pad zeros to feature vectors and test input in case of vocab mismatch
        if X_array.shape[1]>len(self.feat_pos_count):
            self.feat_pos_count = np.pad(self.feat_pos_count, (0,X_array.shape[1]-len(self.feat_pos_count)), 'constant')
            self.feat_neg_count = np.pad(self.feat_neg_count, (0,X_array.shape[1]-len(self.feat_neg_count)), 'constant')
        elif X_array.shape[1]<len(self.feat_pos_count):
            X_array = np.pad(X_array, (0,len(self.feat_pos_count)-X_array.shape[1]), 'constant')

        #Log probability calculation along with the Laplace smoothing factor                
        for files in range(file_nos):
            logProb_pos = np.sum(X_array[files,:] * (np.log(self.feat_pos_count + self.ALPHA)
                                                        - np.log(np.sum(self.feat_pos_count)+self.ALPHA*X_array.shape[1])))
            pos_pred = logProb_pos + np.log(self.prior[+1])

            logProb_neg = np.sum(X_array[files,:] * (np.log(self.feat_neg_count + self.ALPHA) 
                                                        - np.log(np.sum(self.feat_neg_count)+self.ALPHA*X_array.shape[1])))
            neg_pred = logProb_neg + np.log(self.prior[-1])

            if pos_pred > neg_pred:
                pred_Y[files] = +1
            else:
                pred_Y[files] = -1
        return pred_Y

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    print(nb.Eval(test.X, test.Y))
