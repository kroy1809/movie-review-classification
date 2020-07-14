import numpy as np
import collections
import time

from Data_Preprocessing import readData
from DataPreparation import DataPrep


class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.log_prob_pos = []
        self.log_prob_neg = []
        self.prior = {}
        self.train(X, Y)

    def train(self, X, Y):
        # Feature vector creation
        feat_pos = X[(Y == 1), :]
        feat_pos_count = feat_pos.sum(axis=0)
        self.log_prob_pos = np.log(feat_pos_count + self.ALPHA) - np.log(np.sum(feat_pos_count + self.ALPHA))

        feat_neg = X[(Y == 0), :]
        feat_neg_count = feat_neg.sum(axis=0)
        self.log_prob_neg = np.log(feat_neg_count + self.ALPHA) - np.log(np.sum(feat_neg_count + self.ALPHA))

        # Prior calculation
        prior = collections.Counter(Y)
        self.prior = {k: v / total for total in (sum(prior.values()),) for k, v in prior.items()}

    def predict(self, X):
        pos_pred = X.dot(np.transpose(self.log_prob_pos)) + np.log(self.prior[1])
        neg_pred = X.dot(np.transpose(self.log_prob_neg)) + np.log(self.prior[0])

        pred_Y = [1 if pos_pred[row] > neg_pred[row] else 0 for row in range(X.shape[0])]

        return pred_Y

    def accuracy(self, Y_pred, Y_act):
        return np.sum(np.equal(Y_pred, Y_act)) / len(Y_act)


if __name__ == "__main__":
    data = readData("sentiment labelled sentences")
    split = DataPrep(data.data_matrix, data.tags)
    # split.prune_freq(data.dict_word2ID, 1024)
    # split.prune_tfidf(1024)
    start_tr = time.time()
    nb = NaiveBayes(split.data_train, split.tags_train, 0.1)
    end_tr = time.time()


    train_pred = nb.predict(split.data_train)
    print("Training accuracy = ", end="")
    print(nb.accuracy(train_pred, split.tags_train))

    start_valid = time.time()
    print("Validation accuracy = ", end="")
    valid_pred = nb.predict(split.data_valid)
    end_valid = time.time()
    valid_pred_time = end_valid - start_valid
    print(nb.accuracy(valid_pred, split.tags_valid))

    start_tst = time.time()
    print("Test accuracy = ", end="")
    test_pred = nb.predict(split.data_test)
    end_tst = time.time()
    tst_pred_time = end_tst - start_tst
    print(nb.accuracy(test_pred, split.tags_test))

    print("Training time = ", end_tr - start_tr)
    print("Average testing time = ",(tst_pred_time+valid_pred_time)/900) # 900 rows in validation and test
