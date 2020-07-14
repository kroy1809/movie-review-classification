import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

from Data_Preprocessing import readData
from DataPreparation import DataPrep


class KNN:
    def __init__(self, X, Y, K):
        self.knn = KNeighborsClassifier(n_neighbors=K, metric='cosine')
        self.knn.fit(X, Y)

    def predict(self, X):
        return self.knn.predict(X)

    def accuracy(self, Y_pred, Y_act):
        return np.sum(np.equal(Y_pred, Y_act)) / len(Y_act)


if __name__ == "__main__":
    data = readData("sentiment labelled sentences")
    split = DataPrep(data.data_matrix, data.tags)
    # split.prune_freq(data.dict_word2ID, 1024)
    # split.prune_tfidf(1024)

    max_acc = -float('Inf')
    best_k = 1
    tr_time_sum = 0
    valid_time_sum = 0

    # Trying out possible K values to find the model which best fits the validation set
    for K in range(1, 100):
        start_tr = time.time()
        knn = KNN(split.data_train, split.tags_train, K)
        end_tr = time.time()
        tr_time_sum += (end_tr-start_tr) # Training time summed up

        start_valid = time.time()
        valid_pred = knn.predict(split.data_valid)
        end_valid = time.time()
        valid_acc = knn.accuracy(valid_pred, split.tags_valid)
        valid_time_sum+=(end_valid-start_valid)

        # Choosing best K
        if valid_acc > max_acc:
            max_acc = valid_acc
            best_k = K

    print("K = ", best_k, "gives maximum validation accuracy of",max_acc)

    knn1 = KNN(split.data_train, split.tags_train, best_k)

    train_pred = knn1.predict(split.data_train)
    train_acc = knn1.accuracy(train_pred, split.tags_train)
    print("Training accuracy = ", train_acc)

    start_tst = time.time()
    test_pred = knn1.predict(split.data_test)
    end_tst = time.time()
    test_acc = knn1.accuracy(test_pred, split.tags_test)
    tst_pred_time = end_tst - start_tst
    print("Test accuracy = ",test_acc)

    print("Training time = ", tr_time_sum/100) # Training has been carried out for 100 K values
    print("Average testing time = ", (valid_time_sum + tst_pred_time) / (450*101))
    # 450 rows in validation and test, where the validation set has been executed 100 times and test set once
