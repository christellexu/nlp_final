import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sklearn
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from nltk import WordNetLemmatizer
from nltk import stem
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re


# pd.set_option('display.max_rows', None)

#----Importing Data-----
always = pd.read_csv("reviews_always.csv", header = 0,sep ="\t")
tampax = pd.read_csv("reviews_tampax.csv", header = 0,sep ="\t")

#issues with gillette and pantene and oralb
# oralb = pd.read_csv("reviews_oral-b.csv", header = 0, delim_whitespace=True, error_bad_lines=False,names =["id_review", "date", "product_id", "user_rating", "date_other_format", "user_id", "title", "review"])
# oralb = pd.read_csv("reviews_oral-b.csv", header = 0, sep ="\t")

gillette = pd.read_csv("reviews_gillette.csv", header = 0,sep ="\t")
gillette = pd.DataFrame.dropna(gillette)
gillette = gillette[gillette.review.str.contains("http") == False]
gillette = gillette[gillette.review.str.contains("</nobr>") == False]

pantene = pd.read_csv("reviews_pantene.csv", header = 0,sep ="\t")
pantene = pd.DataFrame.dropna(pantene)
pantene = pantene[pantene.review.str.contains("http") == False]
pantene = pantene[pantene.review.str.contains("</nobr>") == False]

#-----Train/Test-----
split = np.random.randn(len(always)) < 0.6667
always_train = always[split]
always_test = always[~split]

# split = np.random.randn(len(oralb)) < 0.6667
# oralb_train = always[split]
# oralb_test = always[~split]


split = np.random.randn(len(tampax)) < 0.6667
tampax_train = tampax[split]
tampax_test = tampax[~split]


split = np.random.randn(len(gillette)) < 0.6667
gillette_train = gillette[split]
gillette_test = gillette[~split]


split = np.random.randn(len(pantene)) < 0.6667
pantene_train = pantene[split]
pantene_test = pantene[~split]

#-----Extract reviews and labels-----

always_review_train = always_train['review']
# oralb_review_train = oralb_train['review']
tampax_review_train = tampax_train['review']
gillette_review_train = gillette_train['review']
pantene_review_train = pantene_train['review']

always_review_test = always_test['review']
# oralb_review_test = oralb_test['review']
tampax_review_test = tampax_test['review']
gillette_review_test = gillette_test['review']
pantene_review_test = pantene_test['review']

always_label_train = always_train['user_rating']
# oralb_label_train = oralb_train['user_rating']
tampax_label_train = tampax_train['user_rating']
gillette_label_train = gillette_train['user_rating']
pantene_label_train = pantene_train['user_rating']

always_label_test = always_test['user_rating']
# oralb_label_test = oralb_test['user_rating']
tampax_label_test = tampax_test['user_rating']
gillette_label_test = gillette_test['user_rating']
pantene_label_test = pantene_test['user_rating']


# #-----remove non letters-----
always_review_train = always_review_train.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
# oralb_review_train = oralb_review_train.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
tampax_review_train = tampax_review_train.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
gillette_review_train = gillette_review_train.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
pantene_review_train = pantene_review_train.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)

always_review_test = always_review_test.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
# oralb_review_test = oralb_review_test.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
tampax_review_test = tampax_review_test.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
gillette_review_test = gillette_review_test.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
pantene_review_test = pantene_review_test.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)


#-----Convert to lowercase and split into individual words-----
always_lower_train = always_review_train.str.lower()
# oralb_lower_train = oralb_review_train.str.lower()
tampax_lower_train = tampax_review_train.str.lower()
gillette_lower_train = gillette_review_train.str.lower()
pantene_lower_train = pantene_review_train.str.lower()

always_lower_test = always_review_test.str.lower()
# oralb_lower_test = oralb_review_test.str.lower()
tampax_lower_test = tampax_review_test.str.lower()
gillette_lower_test = gillette_review_test.str.lower()
pantene_lower_test = pantene_review_test.str.lower()

always_words_train = always_lower_train.str.split()
# oralb_words_train = oralb_lower_train.str.split()
tampax_words_train = tampax_lower_train.str.split()
gillette_words_train = gillette_lower_train.str.split()
pantene_words_train = pantene_lower_train.str.split()

always_words_test = always_lower_test.str.split()
# oralb_words_test = oralb_lower_test.str.split()
tampax_words_test = tampax_lower_test.str.split()
gillette_words_test = gillette_lower_test.str.split()
pantene_words_test = pantene_lower_test.str.split()

#-----Removing Stop Words-----
stop = pd.read_table("stop_words.txt")
stop_words = set(nltk.corpus.stopwords.words('english')) #searching sets in Python is faster than searching lists

always_words_train = always_words_train.apply(lambda x: [item for item in x if item not in stop_words])
# oralb_words_train = oralb_words_train.apply(lambda x: [item for item in x if item not in stop_words])
tampax_words_train = tampax_words_train.apply(lambda x: [item for item in x if item not in stop_words])
gillette_words_train = gillette_words_train.apply(lambda x: [item for item in x if item not in stop_words])
pantene_words_train = pantene_words_train.apply(lambda x: [item for item in x if item not in stop_words])

always_words_test = always_words_test.apply(lambda x: [item for item in x if item not in stop_words])
# oralb_words_test = oralb_words_test.apply(lambda x: [item for item in x if item not in stop_words])
tampax_words_test = tampax_words_test.apply(lambda x: [item for item in x if item not in stop_words])
gillette_words_test = gillette_words_test.apply(lambda x: [item for item in x if item not in stop_words])
pantene_words_test = pantene_words_test.apply(lambda x: [item for item in x if item not in stop_words])


#-----Lemmentization-----
lemmatizer = WordNetLemmatizer()

always_lemm_train = always_words_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
# oralb_lemm_train = oralb_words_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
tampax_lemm_train = tampax_words_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
gillette_lemm_train = gillette_words_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
pantene_lemm_train = pantene_words_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

always_lemm_test = always_words_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
# oralb_lemm_test = oralb_words_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
tampax_lemm_test = tampax_words_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
gillette_lemm_test = gillette_words_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
pantene_lemm_test = pantene_words_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])


always_join_train = always_lemm_train.apply(lambda x: [" ".join(x)])
# oralb_join_train = oralb_lemm_train.apply(lambda x: [" ".join(x)])
tampax_join_train = tampax_lemm_train.apply(lambda x: [" ".join(x)])
gillette_join_train = gillette_lemm_train.apply(lambda x: [" ".join(x)])
pantene_join_train = pantene_lemm_train.apply(lambda x: [" ".join(x)])

always_join_test = always_lemm_test.apply(lambda x: [" ".join(x)])
# oralb_join_test = oralb_lemm_test.apply(lambda x: [" ".join(x)])
tampax_join_test = tampax_lemm_test.apply(lambda x: [" ".join(x)])
gillette_join_test = gillette_lemm_test.apply(lambda x: [" ".join(x)])
pantene_join_test = pantene_lemm_test.apply(lambda x: [" ".join(x)])

#reformat cause I have a list of lists

always_together_train = [item for sublist in always_join_train for item in sublist]
# oralb_together_train = [item for sublist in oralb_join_train for item in sublist]
tampax_together_train = [item for sublist in tampax_join_train for item in sublist]
gillette_together_train = [item for sublist in gillette_join_train for item in sublist]
pantene_together_train = [item for sublist in pantene_join_train for item in sublist]

always_together_test = [item for sublist in always_join_test for item in sublist]
# oralb_together_test = [item for sublist in oralb_join_test for item in sublist]
tampax_together_test = [item for sublist in tampax_join_test for item in sublist]
gillette_together_test = [item for sublist in gillette_join_test for item in sublist]
pantene_together_test = [item for sublist in pantene_join_test for item in sublist]
#is the funky u gonna be an issue


#-----Bag of Words-----
vectorizer = CountVectorizer()
vec = TfidfVectorizer(min_df = 5, max_df= 0.9, sublinear_tf= True, use_idf=True)

always_vectorizer_train = vectorizer.fit_transform(always_together_train)
always_vector_train = always_vectorizer_train.toarray()
vocab_always_train = vectorizer.get_feature_names()
always_tfidf_train = vec.fit_transform(always_together_train)

always_vectorizer_test = vectorizer.fit_transform(always_together_test)
always_vector_test = always_vectorizer_test.toarray()
vocab_always_test = vectorizer.get_feature_names()
always_tfidf_test = vec.transform(always_together_test)


# oralb_vectorizer_train = vectorizer.fit_transform(oralb_together_train)
# oralb_vector_train = always_vectorizer_train.toarray()
# vocab_oralb_train = vectorizer.get_feature_names()
# oralb_tfidf_train = vec.fit_transform(oralb_together_train)


# oralb_vectorizer_test = vectorizer.fit_transform(oralb_together_test)
# oralb_vector_test = always_vectorizer_test.toarray()
# vocab_oralb_test = vectorizer.get_feature_names()
# oralb_tfidf_test = vec.transform(oralb_together_test)


tampax_vectorizer_train = vectorizer.fit_transform(tampax_together_train)
tampax_vector_train = tampax_vectorizer_train.toarray()
vocab_tampax_train = vectorizer.get_feature_names()
tampax_tfidf_train = vec.fit_transform(tampax_together_train)


tampax_vectorizer_test = vectorizer.fit_transform(tampax_together_test)
tampax_vector_test = tampax_vectorizer_test.toarray()
vocab_tampax_test = vectorizer.get_feature_names()
tampax_tfidf_test = vec.transform(tampax_together_test)


gillette_vectorizer_train = vectorizer.fit_transform(gillette_together_train)
gillette_vector_train = gillette_vectorizer_train.toarray()
vocab_gillette_train = vectorizer.get_feature_names()
gillette_tfidf_train = vec.fit_transform(gillette_together_train)


gillette_vectorizer_test = vectorizer.fit_transform(gillette_together_test)
gillette_vector_test = gillette_vectorizer_test.toarray()
vocab_gillette_test = vectorizer.get_feature_names()
gillette_tfidf_test = vec.transform(gillette_together_test)


pantene_vectorizer_train = vectorizer.fit_transform(pantene_together_train)
pantene_vector_train = pantene_vectorizer_train.toarray()
vocab_pantene_train = vectorizer.get_feature_names()
pantene_tfidf_train = vec.fit_transform(pantene_together_train)


pantene_vectorizer_test = vectorizer.fit_transform(pantene_together_test)
pantene_vector_test = pantene_vectorizer_test.toarray()
vocab_pantene_test = vectorizer.get_feature_names()
pantene_tfidf_test = vec.transform(pantene_together_test)

