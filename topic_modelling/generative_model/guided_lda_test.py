#%%
import guidedlda
import pandas as pd
import numpy as np
from itertools import chain
from operator import itemgetter

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

e_keywords = ['emission reduction', 'biodiversity', 'environmental management systems', 'cerez valdez principles', 'carbon', 'fuel', 'greenhouse gas', 'emission']
s_keywords = ['customer health safety', 'quality', 'privacy', 'product labeling']
g_keywords = ['ceo board member', 'esg related compensation', 'board structure type']
all_keywords = e_keywords + s_keywords + g_keywords
data = pd.read_csv('/workspaces/esg-controversy-tracker/dataset/us_equities_news_dataset.csv')['content'].str.lower()
data = data[data == data]
data = data[data.str.contains('|'.join(all_keywords))]
data = data[0:5000]

print("After filtering for keywords, the dataset contains", data.shape)
# Count Vectorizer
vect = CountVectorizer()  
vects = vect.fit_transform(data)

# Select the first five rows from the data set
td = pd.DataFrame(vects.todense()) 
td.columns = vect.get_feature_names()
term_document_matrix = td.T
term_document_matrix.columns = ['Doc '+str(i) for i in range(len(data))]
term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

# Top 25 words 
term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False)

seed_topic_list = [e_keywords, s_keywords, g_keywords]

vocab = set(list(term_document_matrix.index) + list(all_keywords))
word2id = dict((v, idx) for idx, v in enumerate(vocab))

model = guidedlda.GuidedLDA(n_topics=3, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(term_document_matrix.to_numpy(), seed_topics=seed_topics, seed_confidence=0.15)

n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(list(vocab))[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print(f"Topic {i}: {' '.join(topic_words)}")

