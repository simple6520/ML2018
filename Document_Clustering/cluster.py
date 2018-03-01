from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from optparse import OptionParser
import sys
import codecs
import csv
import re

path_name = sys.argv[1]
output_name = sys.argv[2]

with codecs.open(path_name + '/docs.txt', 'r') as f:
    docs = [line.rstrip('\n') for line in f]

with codecs.open(path_name + '/title_StackOverflow.txt', 'r') as f:
    testset = [line.rstrip('\n') for line in f]

with codecs.open(path_name + '/stoplist', 'r') as f:
    stoplist = [line.rstrip('\n') for line in f]

with codecs.open(path_name + '/check_index.csv', 'rb') as f:
    dummy = f.next()
    test = list(csv.reader(f))

docs = filter(None, docs) # remove empty list

for line in range(len(docs)):
    if docs[line][0] == ' ' or docs[line][0] == '\t' or docs[line][0] == '#':
        docs[line] = ''

docs = filter(None, docs)

regex = re.compile('[^a-zA-Z]')
for line in range(len(docs)):
    docs[line] = regex.sub(' ', docs[line])

docs = filter(None, docs) # remove empty list

for line in range(len(docs)):
    if docs[line][0] == ' ' or docs[line][0] == '\t' or docs[line][0] == '#':
        docs[line] = ''

docs = filter(None, docs)

for line in range(len(docs)):
    docs[line] = list(docs[line].split())


for i in range(len(docs)):
    docs[i] = [text.lower() for text in docs[i] if text.lower() not in stoplist and len(text) > 1]
    if len(docs[i]) < 5:
        docs[i] = []

docs = filter(None, docs)

for i in range(len(docs)):
    docs[i] = ' '.join(docs[i])

data = []
temp_merge = ''
for line in range(len(docs)):
    temp_merge = temp_merge + ' ' + docs[line]
    if (line+1) % 15 == 0:
        data.append(temp_merge)
        temp_merge = ''

vectorizer = TfidfVectorizer(max_df=0.25, max_features=50000,
                                  min_df=1, stop_words='english',
                                  use_idf=True)


X = vectorizer.fit_transform(data)
y = vectorizer.transform(testset)


svd = TruncatedSVD(n_components=50, n_iter=50)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)
y = lsa.transform(y)

km = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=100, verbose=True, tol=0.000001)

km.fit(X)

ans = km.predict(y)

r = open(output_name, 'w')
r.write('ID,Ans\n')

for i in range(len(test)):
    if ans[int(test[i][1])] == ans[int(test[i][2])]:
        a = 1
    else:
        a = 0

    r.write(str(i) + ',' + str(a) + '\n')

r.close()
