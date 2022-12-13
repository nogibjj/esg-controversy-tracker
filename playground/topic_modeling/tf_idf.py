#  TOPIC MODELING/TEXT CLASS. SERIES  #
#             Lesson 02.03            #
# TF-IDF in Python with Scikit Learn  #
#               with                  #
#        Dr. W.J.B. Mattingly         #
#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import json
import glob
import re

#%%
def remove_stops(text, stops):
    text = re.sub(r"AC\/\d{1,4}\/\d{1,4}", "", text)
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([i for i in final if not i.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    return (final)

#%%
news_articles = pd.read_csv('/workspaces/esg-controversy-tracker/dataset/us_equities_news_dataset.csv')['content']
news_articles = news_articles[0:100]
news_articles = [x for x in news_articles if x == x]
#%%
# print (cleaned_docs[0])

vectorizer = TfidfVectorizer(
                                lowercase=True,
                                max_features=100,
                                max_df=0.8, #
                                min_df=5,
                                ngram_range = (1,3),
                                stop_words = "english", 
                                smooth_idf=True

                            )

vectors = vectorizer.fit_transform(news_articles)

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()
denselist = dense.tolist()

#%%
all_keywords = []

for description in denselist:
    x=0
    keywords = []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)
print (news_articles[0])
print (all_keywords[0])

#%%
true_k = 3

model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print(terms[i*10:(i+1)*10])
# %%
