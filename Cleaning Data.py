import numpy as np
import pandas as pd
import csv
from bs4 import BeautifulSoup
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




# #-----remove non letters-----
# print always["review"][1]
# test = BeautifulSoup(always["review"][226])
# test = BeautifulSoup(always["review"])
# print test.get_text()

# always['review'] = always['review'].replace(to_replace=r'[\|\\|.|!|-|]', value='', regex=True)
# oralb['review'] = oralb['review'].replace(to_replace=r'[\|\\|.|!|-|]', value='', regex=True)
# tampax['review'] = tampax['review'].replace(to_replace=r'[\|\\|.|!|-|]', value='', regex=True)
# gillette['review'] = gillette['review'].replace(to_replace=r'[\|\\|.|!|-|]', value='', regex=True)
# pantene['review'] = pantene['review'].replace(to_replace=r'[\|\\|.|!|-|]', value='', regex=True)

always['review'] = always['review'].replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
# oralb['review'] = oralb['review'].replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
tampax['review'] = tampax['review'].replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
gillette['review'] = gillette['review'].replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
pantene['review'] = pantene['review'].replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)



#-----Convert to lowercase and split into individual words-----
always['review'] = always.review.str.lower()
# oralb['review'] = oralb.review.str.lower()
tampax['review'] = tampax.review.str.lower()
gillette['review'] = gillette.review.str.lower()
pantene['review'] = pantene.review.str.lower()


























#-----old code-----

# #-----Letters only-----
# always_review = always['review']
# oralb_review = oralb['review']
# tampx_review = tampax['review']
# gillette_review = gillette['review']
# pantene_review = pantene['review']

# names =["id_review", "date", "product_id", "user_rating", "date_other_format", "user_id", "title", "review"]
# oralb = pd.read_csv("reviews_oral-b.csv", header = 0, delim_whitespace=True, error_bad_lines=False,names =["id_review", "date", "product_id", "user_rating", "date_other_format", "user_id", "title", "review"])

