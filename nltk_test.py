from nltk.book import *
fdist1 = FreqDist(text1[:50])
count = 0
for key, value in fdist1.items():
    count += (len(key)*value)


other_count = 0
for i in range(50):
    other_count += len(text1[i])

slice_count = 0
slice_count += len(text1[:50])

import pdb
pdb.set_trace()