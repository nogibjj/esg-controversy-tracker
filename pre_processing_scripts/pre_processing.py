"""
Pre-Processing Script
Function takes as input *only* pandas series objects and performs various pre-processing steps

Pre-processing functions available include
1. Remove Numbers
2. Remove Roman Numbers
3. Tf-Idf Weighting
Restrict the corpus to only words that have tf-idf values more than a specified threshold
4. Standardise whitespace
Replace all whitespace in the text with only one whitespace
5. Limit POS and lemmatize text
Performs two functions
    a. Limit POS limits the text to only the specified POS supplied to the function
    b. Lemmatises all the words in the text

"""

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer
import pandas as pd

porter_stemmer = PorterStemmer()
whitespace_tokenizer = WhitespaceTokenizer()

def remove_numbers(text_series):
    return text_series.str.replace(pat=" \d+", repl=" ", regex=True)

def remove_roman_numbers(text_series):
    pattern = r"\b(?=[MDCLXVIΙ])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})\b\.?"
    return text_series.str.replace(pat=pattern, repl=" ", regex=True)

def apply_tf_idf_weighting(text_series):
    vectorizer = TfidfVectorizer(
                                lowercase=True,
                                ngram_range = (1,1),
                                stop_words = "english"
                            )
    vectors = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names()

    dense = vectors.todense()
    denselist = dense.tolist()
    all_keywords = []
    for description in denselist:
        x=0
        keywords = []
        for word in description:
            if word > 0:
                keywords.append(feature_names[x])
            x=x+1
        all_keywords.append(keywords)
    
    # Change the return type
    return all_keywords

def standardize_whitespace(text_series):
    return text_series.str.replace(pat=r'\s+', repl=" ", regex=True)

def remove_stopwords(text_series, stopwords_list=stopwords.words("english")):
    pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    return text_series.str.replace(pat=pattern, repl=" ", regex=True)

def remove_non_alphanumeric(text_series):
    return text_series.str.replace(pat=r'\W+', repl=" ", regex=True)

def limit_pos(text_series, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in text_series:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


def stem_text(text):
    text = whitespace_tokenizer.tokenize(text)
    text = [porter_stemmer.stem(x) for x in text]
    return ' '.join(text)

def topic_modeling_preprocess(text_series):
    cleaned_text = text_series.str.lower()
    cleaned_text = remove_numbers(cleaned_text)
    cleaned_text = remove_roman_numbers(cleaned_text)
    cleaned_text = limit_pos(cleaned_text)
    cleaned_text = apply_tf_idf_weighting(cleaned_text)
    cleaned_text = [' '.join(x) for x in cleaned_text]
    return pd.Series(cleaned_text)


if __name__ == "__main__":
    data = pd.read_csv('/workspaces/esg-controversy-tracker/dataset/us_equities_news_dataset.csv', nrows=100)['content'].str.lower()
    op = topic_modeling_preprocess(data)
    print(op)



