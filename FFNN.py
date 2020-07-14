import torch
import torch.nn as nn
import torch.optim as optim
import time

from Data_Preprocessing import readData
from DataPreparation import DataPrep


class FFNN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB, DIM_HIDDEN, NUM_CLASSES=2):
        super(FFNN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.DIM_HIDDEN, self.NUM_CLASSES) = (
        VOCAB_SIZE, DIM_EMB, DIM_HIDDEN, NUM_CLASSES)
        self.embed = nn.Embedding(VOCAB_SIZE, DIM_EMB)
        self.hidden = nn.Linear(DIM_EMB, DIM_HIDDEN)
        self.relu = nn.ReLU()
        self.output = nn.Linear(DIM_HIDDEN, NUM_CLASSES)
        self.log_soft = nn.LogSoftmax(dim=0)

        # Initialize weights
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, X, train):
        word_embeds = torch.sum(self.embed(X), dim=0)
        hidden_layer = self.hidden(word_embeds)
        non_linearity = self.relu(hidden_layer)
        output_layer = self.output(non_linearity)
        return self.log_soft(output_layer)


def train_FFNN(X, Y, vocab_size, n_iter):
    num_embedding = 10
    num_hidden = 200
    num_classes = 2
    learning_rt = 1e-3

    mlp = FFNN(X, Y, vocab_size, num_embedding, num_hidden, num_classes)
    optimizer = optim.Adagrad(mlp.parameters(), lr=learning_rt)

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            mlp.zero_grad()

            logProbs = mlp.forward(X[i], train=True)

            y_onehot = torch.zeros(num_classes)
            y_onehot[int(Y[i])] = 1

            loss = torch.neg(logProbs).dot(y_onehot)
            total_loss += loss

            loss.backward()
            optimizer.step()

        print(f"Loss on epoch {epoch} = {total_loss}")
    return mlp


def eval_FFNN(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))


if __name__ == "__main__":
    data = readData("sentiment labelled sentences")
    split = DataPrep(data.data_matrix, data.tags, data.wordlist_ffnn)
    # split.prune_tfidf(1024, True)
    # split.prune_freq(data.dict_word2ID, 1024, True)

    start_tr = time.time()
    mlp = train_FFNN(split.data_tensor_train, split.tags_train, data.nextID, n_iter=50)
    end_tr = time.time()

    print("Training time = ", end_tr - start_tr)

    print("Training accuracy = ", end="")
    eval_FFNN(split.data_tensor_train, split.tags_train, mlp)

    start_test = time.time()
    print("Validation accuracy = ", end="")
    eval_FFNN(split.data_tensor_valid, split.tags_valid, mlp)
    print("Test accuracy = ", end="")
    eval_FFNN(split.data_tensor_test, split.tags_test, mlp)
    end_test = time.time()

    print("Average testing time = ", (end_test-start_test) / 900)
