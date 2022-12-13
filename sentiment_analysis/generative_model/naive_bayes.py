import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

data_path = 
data = pd.read_csv('News_sentiment_Jan2017_to_Apr2021.csv')

# Pre-process Data
def preprocess_data(string):
    removelist = ""
    result = re.sub('','',string)#remove HTML tags
    result = re.sub('https://.*','',result)#remove URLs  
    result = re.sub(r'\W+', ' ', result)#remove non-alphanumeric characters 
    result = result.lower() # changing result into lower case
    return result

data['Title'] = data['Title'].apply(lambda x: preprocess_data(x)) # preprocess the Title


stop_words = set(stopwords.words('english')) # set words we stop on
data['Title'] = data['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# using lemmatizer to grouping together the different inflected forms of a word so they can be analysed as a single item
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st
data['Title'] = data.Title.apply(lemmatize_text)

# Split the data into training and testing data
Title = data['Title'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
# split dataset into 80% train and 20% test parts using train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(Title, encoded_labels, stratify = encoded_labels)


# Building the Naive Bayes Classifier from scratch
## get the frequency of each word appearing in the training set
CountVec = CountVectorizer(max_features = 3000) 
X = CountVec.fit_transform(train_X)
Vocabulary = CountVec.get_feature_names()
X = X.toarray()
## store the word in the dictionary
word_counts = {}
for l in range(2):
    word_counts[l] = defaultdict(lambda: 0)
## store unique words in the corpus in vocab
for i in range(X.shape[0]):
    l = train_Y[i]
    for j in range(len(Vocabulary)):
        word_counts[l][Vocabulary[j]] += X[i][j]
        
        
        
# smmothened function takes the vocabulary and the raw ‘word_counts’ dictionary 
# returns the smoothened conditional probabilities.
def smoothened_conditional_probabilities(n_label_items, Vocabulary, word_counts, word, text_Y):
    a = word_counts[text_Y][word] + 1
    b = n_label_items[text_Y] + len(Vocabulary)
    return math.log(a/b)

# group by label 
def group_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data

# We define the ‘fit_set’ functions for our classifier.
def fit_set(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_label(x, y, labels) # group by label
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors

# We define the ‘predict’ functions for our classifier
## take title(x) and labels(Y) to fitted on and return the number of titles with each sentiment label and the apriori conditional probabilities. 
def predict(n_label_items, Vocabulary, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in Vocabulary: continue
            for l in labels:
                # apply the smoothened function to the probabilities
                log_w_given_l = smoothened_conditional_probabilities(n_label_items, Vocabulary, word_counts, word, l)
                # increasing the probabilities by 1 
                label_scores[l] += log_w_given_l
        # return the predictions on unseen test titles.
        result.append(max(label_scores, key=label_scores.get))
    return result

# Fitting the Model on Training Set and Evaluating Accuracies on the Test Set
labels = [0,1]
n_label_items, log_label_priors = fit_set(train_X,train_Y,labels)
pred = predict(n_label_items, Vocabulary, word_counts, log_label_priors, labels, test_X)
print("Accuracy of prediction on test set : ", accuracy_score(test_Y,pred))