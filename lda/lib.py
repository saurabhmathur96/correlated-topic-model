from os import path
import csv
from random import sample, randint, shuffle
from collections import namedtuple, Counter
from itertools import chain

from numpy import asscalar
from numpy.random import choice
import numpy as np
import numpy.linalg as linalg
from tqdm import trange

Dataset = namedtuple("Dataset", ["X", "y", "size"])

def read_dataset(dirpath):
    index_path = path.join(dirpath, "index.csv")
    with open(index_path, "r") as index_file:
        index = [row for row in csv.reader(index_file, delimiter=",")]
        data = [(open(path.join(dirpath, filename)).read().strip().split(), int(class_label)) for filename, class_label in index]
        X, y = zip(*data)
        return Dataset(np.array(X), np.array(y), len(index))

def subset_data(data, indices):
    X, y, _ = data
    return Dataset(X=np.array([X[i] for i in indices]),
                   y=np.array([y[i] for i in indices]),
                   size=len(indices))

def train_test_split(data, fraction=.7):
    N = data.size
    '''positive_indices, = np.where(data.y == 1) # list(range(N))
    n_positive = np.size(positive_indices, 0)
    
    train_indices = sample(list(positive_indices), int(n_positive*fraction))
    negative_indices,  = np.where(data.y == 0)
    n_negative = np.size(negative_indices, 0)
    train_indices = np.concatenate([train_indices, sample(list(negative_indices), int(n_negative*fraction))])
    # shuffle(indices)
    '''
    indices = list(range(N))
    train_indices = sample(indices, int(fraction*N))
    train_data = subset_data(data, train_indices)
    
    test_indices = frozenset(indices) - frozenset(train_indices)
    test_data = subset_data(data, test_indices)

    return train_data, test_data

def get_random_subset(data, fraction):
    train, _  = train_test_split(data, fraction)
    return train

def lda(X, n_topics=3, alpha=1e-1, beta=1e-3, n_iters=1000):
    document_word_topic = [(i, word, randint(0, n_topics-1)) for i, document in enumerate(X) for word in document]
    n_documents = len(X)
    topics = list(range(n_topics))
    topic_counts = [Counter([topic for document, word, topic in document_word_topic if document == d]) for d in range(n_documents)]
    word_counts = [Counter([word for document, word, topic in document_word_topic if topic == t]) for t in topics]

    vocabulary = set(chain.from_iterable(X))
    vocabulary = sorted(list(vocabulary))
    vocabulary_size = len(vocabulary)

    def compute_sampling_probability(document, topic, word):
        c_t = word_counts[topic][word]
        c_d = topic_counts[document][topic]
        
        c_t_sum = sum(word_counts[topic].values())
        c_d_sum = sum(topic_counts[document].values())

        return ((c_t+beta)*(c_d+alpha)) / ((vocabulary_size*beta + c_t_sum)*(n_topics*alpha + c_d_sum))

    for _ in trange(n_iters):
        # shuffle(document_word_topic)
        for i, (document, word, topic) in enumerate(document_word_topic):
            topic_counts[document][topic] -= 1
            word_counts[topic][word] -= 1
            ps = [compute_sampling_probability(document, t, word) for t in topics]
            total = sum(ps)
            ps = [p/total for p in ps]
            
            topic = asscalar(choice(topics, 1, p=ps))
            document_word_topic[i] = (document, word, topic)
            topic_counts[document][topic] += 1
            word_counts[topic][word] += 1

    features = np.zeros((n_documents, n_topics))
    for document in range(n_documents):
        for topic in range(n_topics):
            numerator = topic_counts[document][topic] + alpha
            denominator = n_topics*alpha + sum(topic_counts[document].values())
            features[document, topic] = numerator / denominator
    
    return features, document_word_topic, topic_counts, word_counts


OptConfig = namedtuple("OptConfig", ["update_step", "hyper_parameters", "stopping_criterion", "n_steps", "learning_rate_decay"])


def optimize(w, X, y, update_step, hyper_parameters, stopping_criterion, n_steps, learning_rate_decay=None):
    for step in range(n_steps):

        w_next = update_step(X, y, w, **hyper_parameters)
        if "learning_rate" in hyper_parameters and learning_rate_decay:
            lr = hyper_parameters["learning_rate"]
            hyper_parameters["learning_rate"] = lr / (1 + lr*learning_rate_decay*step)

        if stopping_criterion(w, w_next):
            return w_next
        
        w = w_next

    return w

def sigmoid(x):
    negative = x<0
    x[~negative] = 1/(1+np.exp(-x[~negative]))
    x[negative] = np.exp(x[negative])/(1+np.exp(x[negative]))
    return x

def add_bias(X):
    n_examples = np.size(X, 0)
    return np.hstack([np.ones((n_examples, 1)), X])

def compute_precision(X, y, w, alpha):
    n_features = np.size(X, 1)
    y_hat = sigmoid(w.dot(X.T))
    R = np.diag(y_hat*(1 - y_hat))
    precision = alpha * np.eye(n_features) + X.T.dot(R).dot(X)
    return precision + 1e-9*np.eye(n_features)

class LogisticRegressionClassifier:
    def __init__(self, w_mean=None, w_precision=None, optconfig=None):
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.optconfig = optconfig
    


    def fit(self, X, y, optconfig=None):
        if not optconfig:
            if not self.optconfig:
                raise ValueError('No optconfig value provided and no default set.')
            else:
                optconfig = self.optconfig

        X = add_bias(X)
        n_features = np.size(X, 1)
        initial_w = np.zeros(n_features) + 1e-9 # self.w_mean if self.w_mean is not None else np.zeros(n_features) + 1e-9 
        w = optimize(w=initial_w,
                     X=X, y=y,
                     update_step=optconfig.update_step, 
                     hyper_parameters=optconfig.hyper_parameters,
                     stopping_criterion=optconfig.stopping_criterion,
                     n_steps=optconfig.n_steps,
                     learning_rate_decay=optconfig.learning_rate_decay)
        alpha = optconfig.hyper_parameters['alpha']
        precision = compute_precision(X, y, w, alpha)
        self.w_mean = w
        self.w_precision = precision 
        return self
        
    def predict(self, X):
        X = add_bias(X)
        s_a_squared = np.sum(linalg.solve(self.w_precision, X.T).T * X, axis=1) # Efficient way to compute A^{-1}B
        mu_a = self.w_mean.dot(X.T)
        return (sigmoid(mu_a / np.sqrt(1 + s_a_squared*np.pi*.125)) >= .5).astype(int)


def newton_raphson_update(X, y, w, alpha):
    n_features = np.size(X, 1)
    y_hat = sigmoid(w.dot(X.T))
    R = np.diag(y_hat*(1 - y_hat))
    hessian = alpha * np.eye(n_features) + X.T.dot(R).dot(X) + 1e-9*np.eye(n_features)
    gradient = X.T.dot(y_hat - y) + alpha*w
    update = linalg.solve(hessian, gradient) # Efficient way to compute A^{-1}B
    return w - update

def stopping_criterion(w, w_next):
    return linalg.norm(w_next - w, ord=2) / linalg.norm(w, ord=2) < 1e-3



newton_raphson = OptConfig(
    update_step=newton_raphson_update,
    stopping_criterion=stopping_criterion,
    hyper_parameters=dict(alpha=0.01),
    n_steps=150,
    learning_rate_decay=None
)
def accuracy_score(prediction, ground_truth):
    return np.average(prediction == ground_truth)
