import os
import pdb

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import VietnameseProcess


def get_item(content, item):
    sentence = VietnameseProcess(content)
    sentence.progress()
    return [{
        'feature': sentence.sentence,
        'target': 0 if item == '-1' else 1
    }]


def load_dataset(path):
    ds = []
    items = os.listdir(path)
    for item in items:
        item_path = '%s/%s' % (path, item)
        for file in os.listdir(item_path):
            file_path = '%s/%s' % (item_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                ds = ds + get_item(f.read(), item)

    df_test = pd.DataFrame(ds)
    return df_test


def vectorize(df_test):
    v = TfidfVectorizer()
    train_vectors = v.fit_transform(df_test)
    return train_vectors


# Đọc dataset và preprocess
df = load_dataset('dataset1//test')

# vector hóa
vector = vectorize(df.feature)

# Chạy KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(vector)

print('Negative samples accuracy: ' + str(metrics.accuracy_score(df.target[:6806], kmeans.labels_[:6806]) * 100))
print('Positive samples accuracy: ' + str(metrics.accuracy_score(df.target[6806:], kmeans.labels_[6806:]) * 100))
print('Model accuracy: ' + str(metrics.accuracy_score(df.target, kmeans.labels_) * 100))

pdb.set_trace()