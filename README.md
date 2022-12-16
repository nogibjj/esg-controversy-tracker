# Sentiment Analysis of Indian News Headlines
## 1 Introduction 
Here is for IDS 703 final project for Aditya John, Scott Lai, and Pooja Kabber. We focus on the research on ESG controversy trackers using NLP. We chose to use both the generative model and the discriminative model to do the sentiment analysis for the ESG news headlines in our study.

## 2 Data
### 2.1 Description
The dataset for this project is the Indian Financial News Headlines dataset from kaggle. It consists of 200288 unique data points with the dataset having 54% negative and 46% positive posts. The dataset also contains confidence thresholds for the sentiment score it has assigned to either positive or negative sentiment. 

For this project, we only use a subset of the dataset. First, we filter only to include articles with a confidence greater than 99%. Post which we shuffle the dataset and use 25k positive and negative samples. Our final dataset therefore contains of 50k datapoint with 50% positive and 50% negative posts. Our reasons for doing this is twofold - first is to improve the quality of the dataset, by applying the confidence threshold. Additionally, by restricting the dataset to only 50k rows, we reduce the computational requirements for the discriminative model. 


### 2.2 Pre-Processing
For the generative model, cleaning and pre-processing the dataset is important to improve the results. Because the model used is a bag-of-words model, reducing the noise in the dataset will help the model learn a more appropriate probability distribution. 

We perform the following pre-processing steps
### 2.2.1 Stopword Removal
Words that frequently occur in the corpus that dont convey any meaning are considered stopwords. For example, words like “the”, “and”, “an” etc. which dont have any inherent meaning but are required to form grammatical sentences are considered stopwords. Removing them for the corpus before training, will allow the model to focus more on words that contribute to the task at hand, in this case sentiment analysis. 

### 2.2.2 Non-alphanumeric character removal (Incomplete)
Similar to stopword removal, this also follows through to do

### 2.2.3. Stemming
Stemming is the process of reducing words in the sentence to their root stem. For example, for words like “larger” and “largest”, the root stem is “large”. Performing stemming on the dataset helps reduce the vocabulary as different inflections of the word get merged into one. 

Typically, lemmatization works better than stemming as it focuses on the “lemma” of the word rather than the “stem”. For example, after stemming the words “be”, “is” and “are” remain different, whereas when applying lemmatization, they get merged to the same word “be”. Lemmatization would further reduce the dataset as compared to stemming. However, our reason for stemming was primarily due to resource constrains as lemmatization is a lot more computationally intensive. 

### 2.2.4. Removing Numbers
While this is not a usual pre-processing step, numbers do not add a lot of meaning when it comes to sentiment analysis. For example, consider the two headlines “Sensex , Nifty continue to struggle ; BHEL , Maruti rise 1 %” and “2017 could be cyclically a very good year for largecap IT : Hiren Ved , Alchemy Capital”. In both cases, the numbers cannot be used to determine the sentiment score. However, it is possible that the numbers could bias the model towards one or the other sentiment. For example, most of the headlines during 2020 would be negative given the market crash and negative macro environment. Therefore, 2020 would have have a higher negative probability when compared to positive probability. Any sentence that has the number 2020 would be biased towards having a negative sentiment. 


## 2.3 Synthetic Data Generation
Synthetic data refers to data created by the generative model. Post training, the model learns the posterior probability of the words for each class (positive and negative). By sampling the distribution of each class (positive and negative) we can generate word distributions of a given length. Because of the “naive” assumption of the model, i.e. that each word is independent of the other, each word is sampled independently. Therefore, it is very likely that the sentence generated do not follow any grammatical rules, but simply a set of words that follow the probability distribution. 



# 3 Model
## 3.1 Generative 
For sentiment analysis of text data, including news headlines, Naive Bayes is a generative model that can be used. Its foundation is the notion that predictions about the class (such as positive, negative, or neutral) of a given text can be made using probabilities.
For sentiment analysis, news headlines are categorized as having a positive, negative, or neutral sentiment, and this information is used to train the model. The algorithm learns the likelihood of various words or phrases occurring in headlines with each sentiment class using this training data. 
The model determines the likelihood that a given headline belongs to each sentiment class during the prediction phase based on the words and phrases it includes. The sentiment of the headline is then predicted to be the class with the highest likelihood.
The "naive" assumption, one of the fundamental elements of the naive Bayes model, states that all features (in this case, the words and phrases in the headline) are independent of one another. This presumption makes the computations easier and makes it possible to train and operate the model quickly, but it might not always be true in actual use.
Overall, naive Bayes is a straightforward and efficient technique for doing sentiment analysis, and it has been widely applied for this purpose in a variety of scenarios, including the analysis of headlines from Indian news sources.




## 3.2 Discriminative
For the discriminative model, we are approached by using the BERT approach to do the sentimental modeling analysis. BERT is a transformer-based model that has shown amazing results in various NLP tasks over the past years. We are fine-tuning the model using the Hugging face and applying the pre-train model in our ESG test set. 



# 4. Results
## 4.1 Generative Model
## 4.2 Discriminative Model
Generative model results
real data - 87.176%
synthetic data - 91.288%
Discriminative model results
real data - 92.01%
synthetic data - 49.40%

# 5 Conclusion



1. Generative model results
    - real data - 87.176%
    - synthetic data -  91.288%

2. Discriminative model results
    - real data - 92.01%
    - synthetic data - 49.40%
