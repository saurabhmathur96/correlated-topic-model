from lib import read_dataset, train_test_split, lda, Dataset, get_random_subset
from lib import LogisticRegressionClassifier, newton_raphson, accuracy_score
from itertools import chain
import numpy as np
import math
import matplotlib.pyplot as plt

def count_vectorize(X, vocabulary=None):
    if vocabulary is None:
        vocabulary = sorted(list(set(chain.from_iterable(X))))
    N = len(X)
    V = len(vocabulary)
    counts = np.zeros((N, V))
    for i, document in enumerate(X):
        for word in document:
            try:
                j = vocabulary.index(word)
            except ValueError:
                continue
            counts[i, j] += 1

    return counts, vocabulary


data = read_dataset("pp4data/20newsgroups")
train, test = train_test_split(data, fraction=2/3)
train_X, vocabulary = count_vectorize(train.X)
test_X, _ = count_vectorize(test.X, vocabulary=vocabulary)
train = Dataset(X=train_X, y=train.y, size=train.size)
test = Dataset(X=test_X, y=test.y, size=test.size)
train_set_sizes = range(2, train.size, math.ceil(train.size/30))
bow_scores = list()
bow_sd = list()
for train_size in train_set_sizes:
    scores = list()
    for _ in range(30):
        subset = get_random_subset(
            train, fraction=train_size/data.size)
        scores.append(accuracy_score(LogisticRegressionClassifier(optconfig=newton_raphson)
                            .fit(subset.X, subset.y).predict(test.X), test.y))
    bow_scores.append(np.average(scores))
    bow_sd.append(np.std(scores))


X = np.loadtxt("lda-features.txt")
data = Dataset(X=X, y=data.y, size=data.size)
train, test = train_test_split(data, fraction=2/3)

train_set_sizes = range(2, train.size, math.ceil(train.size/30))
lda_scores = list()
lda_sd = list()
for train_size in train_set_sizes:
    scores = list()
    for _ in range(30):
        subset = get_random_subset(
            train, fraction=train_size/data.size)
        scores.append(accuracy_score(LogisticRegressionClassifier(optconfig=newton_raphson)
                            .fit(subset.X, subset.y).predict(test.X), test.y))
    lda_scores.append(np.average(scores))
    lda_sd.append(np.std(scores))

plt.title("Learning Curves for Logistic Regression on raw data and on LDA Features")
plt.errorbar(train_set_sizes, bow_scores, yerr=bow_sd)
plt.errorbar(train_set_sizes, lda_scores, yerr=lda_sd)
plt.ylabel("Accuracy Score (on test data)")
plt.xlabel("Train Set Size")
plt.legend(["Bag of Words", "LDA"])
plt.savefig("Figure_1.png")
