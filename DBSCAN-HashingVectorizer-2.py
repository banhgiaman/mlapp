import os
import pdb

import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import HashingVectorizer

from preprocess import VietnameseProcess


def get_item(content, item):
    sentence = VietnameseProcess(content)
    sentence.progress_DBSCAN()
    return [{
        'feature': sentence.sentence,
        'target': -1 if item == 'neg' else 0
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
    v = HashingVectorizer()
    train_vectors = v.fit_transform(df_test)
    return train_vectors


# Đọc dataset và preprocess
df = load_dataset('dataset2')

# vector hóa
vector = vectorize(df.feature)

# Chạy thuật toán DBSCAN
db = DBSCAN(eps=0.8, min_samples=55).fit(vector)
print('Negative samples accuracy: ' + str(metrics.accuracy_score(df.target[:5000], db.labels_[:5000]) * 100))
print('Positive samples accuracy: ' + str(metrics.accuracy_score(df.target[5000:], db.labels_[5000:]) * 100))
print('Model accuracy: ' + str(metrics.accuracy_score(df.target, db.labels_) * 100))
pdb.set_trace()