import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import VietnameseProcess

def get_item(content, item):
    sentence = VietnameseProcess(content)
    sentence.progress()
    return [{
        'feature': sentence.sentence,
        'target': item
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


df = load_dataset('data_train')
vector = vectorize(df.feature)

# Chạy thử KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(vector)


import pdb
pdb.set_trace()
