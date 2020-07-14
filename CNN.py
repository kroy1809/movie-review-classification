import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim

from imdb import IMDBdata


class CNN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, POOL_DIM=50, N_FILTERS=100, DIM_EMB=10, NUM_CLASSES=2):
        super(CNN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        # TODO: Initialize parameters.
        self.embed = nn.Embedding(VOCAB_SIZE, DIM_EMB)
        self.conv_uni = nn.Conv1d(1, N_FILTERS, (1, DIM_EMB))
        self.conv_bi = nn.Conv1d(1, N_FILTERS, (2, DIM_EMB))
        self.conv_tri = nn.Conv1d(1, N_FILTERS, (3, DIM_EMB))

        self.max_pooling = nn.AdaptiveMaxPool1d(POOL_DIM)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(N_FILTERS * POOL_DIM * 3, NUM_CLASSES)
        self.log_soft = nn.LogSoftmax(dim=0)

        # Initalize weights (Glorot and Bengio 2010)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.conv_uni.weight)
        nn.init.xavier_uniform_(self.conv_bi.weight)
        nn.init.xavier_uniform_(self.conv_tri.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, X, train=False):
        # TODO: Implement forward computation.
        embedded = self.embed(X).unsqueeze(0).unsqueeze(0)

        conv_1 = self.conv_uni(embedded).squeeze(3)
        conv_2 = self.conv_bi(embedded).squeeze(3)
        conv_3 = self.conv_tri(embedded).squeeze(3)

        mx_pool1 = self.relu(self.max_pooling(conv_1))
        mx_pool2 = self.relu(self.max_pooling(conv_2))
        mx_pool3 = self.relu(self.max_pooling(conv_3))

        cat = torch.cat((mx_pool1, mx_pool2, mx_pool3), dim=1).view(1, -1)

        return self.log_soft(self.fc(cat).squeeze(0))


def Eval_CNN(X, Y, cnn):
    num_correct = 0
    for i in range(len(X)):
        logProbs = cnn.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))


def Train_CNN(X, Y, vocab_size, n_iter):
    num_classes = 2
    learning_rt = 1e-4

    print("Start Training!")
    cnn = CNN(X, Y, vocab_size)
    # TODO: initialize optimizer.
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rt)

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            # TODO: compute gradients, do parameter update, compute loss.
            cnn.zero_grad()

            logProbs = cnn.forward(X[i], train=True)

            y_onehot = torch.zeros(num_classes)
            y_onehot[int(Y[i])] = 1

            loss = torch.neg(logProbs).dot(y_onehot)
            total_loss += loss

            loss.backward()
            optimizer.step()
        print(f"loss on epoch {epoch} = {total_loss}")
    return cnn


if __name__ == "__main__":
    data = readData("sentiment labelled sentences")
    split = DataPrep(data.data_matrix, data.tags, data.wordlist_ffnn)

    cnn = Train_CNN(split.data_tensor_train, (split.tags_train + 1.0) / 2.0, data.nextID, n_iter=25)
    print("Validation accuracy = ", end="")
    Eval_CNN(split.data_tensor_valid, (split.tags_valid + 1.0) / 2.0, cnn)
    print("Test accuracy = ", end="")
    Eval_CNN(plit.data_tensor_test, (split.tags_test + 1.0) / 2.0, cnn)