import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata


class Perceptron:
    def __init__(self, X, Y, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        # TODO: Initalize parameters
        self.weights = np.zeros(X.shape[1] + 1)
        self.weights_avg = np.zeros(X.shape[1] + 1)
        self.counter = 1
        self.Train(X, Y)

    def ComputeAverageParameters(self):
        # TODO: Compute average parameters (do this part last)
        self.weights -= self.weights_avg / self.counter

    def Train(self, X, Y):
        print("Start training!")
        file_nos = X.shape[0]
        print("Epoch progress: ", end="")
        # TODO: Estimate perceptron parameters
        for epoch in range(self.N_ITERATIONS):
            print(epoch, end=" ")
            for file in range(file_nos):
                activation = X[file, :].dot(self.weights[1:]) + self.weights[0]
                if Y[file] * activation <= 0:
                    self.weights[1:] += Y[file] * X[file, :]
                    self.weights[0] += Y[file]
                    self.weights_avg[1:] += self.counter * Y[file] * X[file, :]
                    self.weights_avg[0] += self.counter * Y[file]
                self.counter += 1
        print("\nTraining complete!")

    def Predict(self, X):
        # TODO: Implement perceptron classification
        return np.where(X.dot(self.weights[1:]) + self.weights[0] > 0, 1, -1)

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()


if __name__ == "__main__":
    data = readData("sentiment labelled sentences")
    split = DataPrep(data.data_matrix, data.tags)
    # split.prune_tfidf(1024, True)
    # split.prune_freq(data.dict_word2ID, 1024, True)

    ptron = Perceptron(split.data_train, split.tags_train, N_ITERATIONS=50)
    ptron.ComputeAverageParameters()

    print("Testing on dev set:", ptron.Eval(split.data_valid, split.tags_valid))

    print("Testing on test set:", ptron.Eval(split.data_test, split.tags_test))

    # Print out the 20 most positive and 20 most negative words
    pos_word_id = list(reversed(np.argsort(ptron.weights[1:])))
    print(pos_word_idx)
    pos_word_wts = [ptron.weights[pos_word_id[x] + 1] for x in range(20)]
    pos_words = [train.vocab.GetWord(pos_word_id[x]) for x in range(20)]

    print("Top 20 positive words with their weights:")
    for words, wts in zip(pos_words, pos_word_wts):
        print(f'{words:>18}  {wts:>20}')

    neg_word_id = list(np.argsort(ptron.weights[1:]))
    neg_word_wts = [ptron.weights[neg_word_id[x] + 1] for x in range(20)]
    neg_words = [train.vocab.GetWord(neg_word_id[x]) for x in range(20)]

    print("Top 20 negative words with their weights:")
    for words, wts in zip(neg_words, neg_word_wts):
        print(f'{words:>18}  {wts:>20}')