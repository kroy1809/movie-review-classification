import os

from nltk import word_tokenize
from collections import Counter
from scipy.sparse import csr_matrix

import numpy as np

np.random.seed(1)

class readData:
    def __init__(self, directory):
        self.dict_word2ID = {}  # Map to store key = word and value = ID
        self.dict_ID2word = {}  # Map to store key = ID and value = word
        self.nextID = 0  # For assigning words with IDs in the maps declared above

        row_cnt = 0  # Counter variable for tagging each sentence differently

        self.tags = []  # List to store the tags associated with each sentence
        matrix_row = []  # List to store the matrix row indices
        matrix_col = []  # List to store the matrix column indices
        matrix_val = []  # List to store the matrix values

        self.wordlist_ffnn = []

        files = os.listdir("%s" % directory)  # Lists all the files in the directory passed as an argument
        if "readme.txt" in files: files.remove("readme.txt")  # Ignore readme.txt file if present in the directory

        for i in range(len(files)):
            if files[i].endswith(".txt"):  # Only consider .txt files for reading
                f = open("%s/%s" % (directory, files[i]))
                for lines in f.readlines():
                    wordlist = [self.getwordID(w.lower())
                                for w in word_tokenize(str(lines.split('\t')[0]))]
                    self.wordlist_ffnn.append(wordlist)
                    wordCounts = Counter(wordlist)  # Creates a map with the counts for each word per sentence
                    for (wordid, count) in wordCounts.items():
                        matrix_row.append(row_cnt)
                        matrix_col.append(wordid)
                        matrix_val.append(count)
                    self.tags.append(int(lines.split('\t')[1]))  # Creates a vector of the tags for the sentences
                    # For testing, uncomment the lines below
                    # if row_cnt<5: print(wordCounts)
                    # if row_cnt == 4:
                    #     print(self.dict_word2ID)
                    #     print(tags)
                    row_cnt += 1

        # Create a sparse matrix with counts of each word for the sentences
        self.data_matrix = csr_matrix((matrix_val, (matrix_row, matrix_col)),
                                      shape=(max(matrix_row) + 1, max(matrix_col) + 1))
        self.tags = np.asarray(self.tags)

    # Function to associate each word with an ID
    def getwordID(self, word):
        if word not in self.dict_word2ID:
            self.dict_word2ID[word] = self.nextID
            self.dict_ID2word[self.dict_word2ID[word]] = word
            self.nextID += 1
        return self.dict_word2ID[word]


if __name__ == "__main__":
    reader = readData("sentiment labelled sentences")  # Change the folder name if saved with a different name
    x = reader.data_matrix.sum(axis=1)



