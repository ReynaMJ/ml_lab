import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# one topic only, split into train and test
train = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  categories=['sci.space'], remove=('headers','footers','quotes'))

vec = CountVectorizer(max_features=5000)
train_counts = np.asarray(vec.fit_transform(train.data).sum(axis=0))[0]   # word counts in train
test_counts  = np.asarray(vec.transform(test.data).sum(axis=0))[0]        # word counts in test
N, V = train_counts.sum(), len(train_counts)

for alpha in [1, 2, 5, 50]:        # alpha = 1 is MLE (flat prior)
    p = (train_counts + alpha - 1) / (N + V * (alpha - 1))     # MAP estimate
    p = np.maximum(p, 1e-12)                                    # avoid log(0)
    loglik = (test_counts * np.log(p)).sum()                    # how well p explains test data
    print("MLE" if alpha == 1 else f"MAP alpha={alpha}", "-> test log-likelihood:", round(loglik))