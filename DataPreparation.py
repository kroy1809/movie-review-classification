import numpy as np
from nltk.corpus import stopwords
import torch

from Data_Preprocessing import readData


class DataPrep:
    def __init__(self, X, Y, X_wordlist=None, train_size=0.7, valid_size=0.15):
        # Random shuffle
        self.index = np.arange(X.shape[0])
        np.random.shuffle(self.index)
        X = X[self.index, :]
        Y = Y[self.index]
        self.wordlist = X_wordlist
        self.train_size = train_size
        self.valid_size = valid_size
        if X_wordlist is not None:
            X_tensor = [torch.LongTensor(X_wordlist[i]) for i in self.index]

        # Split data in training, validation and test sets
        self.data_train = X[:int(self.train_size * X.shape[0]), :]
        self.data_valid = X[int(self.train_size * X.shape[0]):int((self.train_size + self.valid_size) * X.shape[0]), :]
        self.data_test = X[int((self.train_size + self.valid_size) * X.shape[0]):, :]

        # Split tags in training, validation and test sets
        self.tags_train = Y[:int(self.train_size * Y.shape[0])]
        self.tags_valid = Y[int(self.train_size * Y.shape[0]):int((self.train_size + self.valid_size) * Y.shape[0])]
        self.tags_test = Y[int((self.train_size + self.valid_size) * X.shape[0]):]

        if X_wordlist is not None:
            self.data_tensor_train = X_tensor[:int(self.train_size * len(X_tensor))]
            self.data_tensor_valid = X_tensor[
                                     int(self.train_size * len(X_tensor)):int(
                                         (self.train_size + self.valid_size) * len(X_tensor))]
            self.data_tensor_test = X_tensor[int((self.train_size + self.valid_size) * len(X_tensor)):]

    def prune_tfidf(self, num_feat=1024, prune_tensor=False):
        tf = self.data_train / np.sum(self.data_train, axis=1) # Calculate Term Frequency
        # Calculate Inverse Document Frequency (with smoothing)
        idf = np.array([np.log(self.data_train.shape[0] / (1 + self.data_train[:, col].count_nonzero()))
                        for col in range(self.data_train.shape[1])])
        tfidf = np.amax(np.multiply(tf, idf), axis=0) # Obtain TF-IDF scores for each word (Maximum TF-IDF in corpus)
        wid_sorted = np.array(np.argsort(tfidf)).flatten() # Get IDs for the words sorted by their TF-IDF scores
        wid_exists = wid_sorted[-num_feat:]

        # Prune all sets the feature vectors
        self.data_train = self.data_train[:, wid_exists]
        self.data_valid = self.data_valid[:, wid_exists]
        self.data_test = self.data_test[:, wid_exists]

        # Prune the tensors in case FFNN is the classifier
        if prune_tensor:
            new_wordlist = []
            for sen in self.index:
                tmp_wordlist = []
                for word in range(len(self.wordlist[sen])):
                    if self.wordlist[sen][word] in wid_exists:
                        tmp_wordlist.append(self.wordlist[sen][word])
                new_wordlist.append(torch.LongTensor(tmp_wordlist))

            self.data_tensor_train = new_wordlist[:int(self.train_size * len(new_wordlist))]
            self.data_tensor_valid = new_wordlist[
                                     int(self.train_size * len(new_wordlist)):int(
                                         (self.train_size + self.valid_size) * len(new_wordlist))]
            self.data_tensor_test = new_wordlist[int((self.train_size + self.valid_size) * len(new_wordlist)):]

    def prune_freq(self, words_idx, num_feat=1024, prune_tensor=False):
        # List of stopwords
        stop_words_ids = [words_idx[wrd] for wrd in list(stopwords.words('english')) if wrd in words_idx]
        freq = np.sum(self.data_train, axis=0) # Calculate frequency of all words across the entire corpus
        wid_freq = np.array(np.argsort(freq)).flatten() # Obtain IDs for words sorted by frequency
        wid_exists = wid_freq[np.isin(wid_freq, stop_words_ids, invert=True)][-num_feat:] # Remove stopwords

        # Prune all sets the feature vectors
        self.data_train = self.data_train[:, wid_exists]
        self.data_valid = self.data_valid[:, wid_exists]
        self.data_test = self.data_test[:, wid_exists]

        # Prune the tensors in case FFNN is the classifier
        if prune_tensor:
            new_wordlist = []
            for sen in self.index:
                tmp_wordlist = []
                for word in range(len(self.wordlist[sen])):
                    if self.wordlist[sen][word] in wid_exists:
                        tmp_wordlist.append(self.wordlist[sen][word])
                new_wordlist.append(torch.LongTensor(tmp_wordlist))

            self.data_tensor_train = new_wordlist[:int(self.train_size * len(new_wordlist))]
            self.data_tensor_valid = new_wordlist[
                                     int(self.train_size * len(new_wordlist)):int(
                                         (self.train_size + self.valid_size) * len(new_wordlist))]
            self.data_tensor_test = new_wordlist[int((self.train_size + self.valid_size) * len(new_wordlist)):]

# if __name__ == "__main__":
#     data = readData("sentiment labelled sentences")
#     split = DataPrep(data.data_matrix, data.tags, data.wordlist_ffnn)
#     split.prune_tfidf(1024)
