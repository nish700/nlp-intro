# --------------
# Importing Necessary libraries
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the 20newsgroups dataset
newsgroup_train = fetch_20newsgroups(subset='train')
pprint(list(newsgroup_train))

#Create a list of 4 newsgroup and fetch it using function fetch_20newsgroups
target_newsgroups = ['sci.space','rec.autos','talk.politics.mideast','talk.politics.guns']
newsgroups_cust_train = fetch_20newsgroups(subset='train', categories = target_newsgroups)

#Use TfidfVectorizer on train data 
vectorizer = TfidfVectorizer(stop_words='english')

vectors_train = vectorizer.fit_transform(newsgroups_cust_train.data)

# find out the Number of Non-Zero components per sample.
non_zero = vectors_train.nnz / float(vectors_train.shape[0])

print('Average of {0} non zero components by sample in more than 30000 dimensional space is:'.format(non_zero))

#Use TfidfVectorizer on test data and apply Naive Bayes model and calculate f1_score.
# load the test data
newsgroups_test = fetch_20newsgroups(subset='test', categories = target_newsgroups)

vectors_test = vectorizer.transform(newsgroups_test.data)

# initialise NB
nb = MultinomialNB(alpha=0.01)
# fit on train data
nb.fit(vectors_train, newsgroups_cust_train.target)

# predict
pred = nb.predict(vectors_test)

# calculate the f1 score
f_score = f1_score(newsgroups_test.target, pred, average = 'macro')
print('f1 score is :', f_score)

#Print the top 20 news category and top 20 words for every news category

# function for top 20 words from category
def show_top20(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    print('Top 20 categories for the selected features are:')
    for i , category in enumerate(categories):
        top20 = np.argsort(classifier.coef_[i])[-20:]
        print('%s : %s' % (category , " ".join(feature_names[top20])))


