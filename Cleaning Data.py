import numpy as np
import pandas as pd
import csv
import sklearn
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk import WordNetLemmatizer
from nltk import stem
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltkx
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


always_review = always['review']
# oralb_review = oralb['review']
tampax_review = tampax['review']
gillette_review = gillette['review']
pantene_review = pantene['review']



# #-----remove non letters-----
always_review = always_review.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
# oralb_review = oralb_review.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
tampax_review = tampax_review.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
gillette_review = gillette_review.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
pantene_review = pantene_review.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)



#-----Convert to lowercase and split into individual words-----
always_lower = always_review.str.lower()
# oralb_lower = oralb_review.str.lower()
tampax_lower = tampax_review.str.lower()
gillette_lower = gillette_review.str.lower()
pantene_lower = pantene_review.str.lower()


always_words = always_lower.str.split()
# oralb_words = oralb_lower.str.split()
tampax_words = tampax_lower.str.split()
gillette_words = gillette_lower.str.split()
pantene_words = pantene_lower.str.split()

#-----Removing Stop Words-----
stop = pd.read_table("stop_words.txt")
stop_words = set(nltk.corpus.stopwords.words('english')) #searching sets in Python is faster than searching lists

always_words = always_words.apply(lambda x: [item for item in x if item not in stop_words])
# oralb_words = oralb_words.apply(lambda x: [item for item in x if item not in stop_words])
tampax_words = tampax_words.apply(lambda x: [item for item in x if item not in stop_words])
gillette_words = gillette_words.apply(lambda x: [item for item in x if item not in stop_words])
pantene_words = pantene_words.apply(lambda x: [item for item in x if item not in stop_words])


#-----Stemming & Lemmentization-----
lemmatizer = WordNetLemmatizer()

always_lemm = always_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
# oralb_lemm = oralb_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
tampax_lemm = tampax_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
gillette_lemm = gillette_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
pantene_lemm = pantene_words.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])








#-----Questions----
#does it matter which stopwords list we use, do we have to use the one provided

