import re
import sys
import requests
import pandas as pd
import numpy as np

!pip install mwparserfromhell
import mwparserfromhell


from spacy.en import STOP_WORDS
from spacy.en import English
nlp = English()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

## Loading pickle file, having data from mongoDB
ml_bs_df = pd.read_pickle('/home/jovyan/Jupyter_Repository/Practice_exercise/Project_4-wkiAPI-/Output/ml_bs_df.p')
## creating new column for Category name1:1636 = 'Machine Learning, 1637: = 'Business Software
ml_bs_df['master_category'] = 'Machine learning'
ml_bs_df['master_category'][1637:] = 'Business software' 
## deleting rows with duplicate text
ml_bs_df.drop('_id', axis = 1, inplace=True)
ml_bs_df.drop_duplicates(inplace=True)
ml_bs_df.shape

##using mwparserfromhell to clear the jason format
#ml_bs_df['clean_content'] = ml_bs_df['content'].apply(lambda x: mwparserfromhell.parse(x).strip_code().replace('\n', ' '))
## defining clean function, using regex, lemma and stopwords from spacey
def cleaner(text):
    #text = re.sub('&#39;','',text).lower()
    text = re.sub('[\d]','',text)
    text = re.sub('\\ufeff', '', text)
    text = re.sub('[^a-z ]','', text)
    text = ' '.join(i.lemma_ for i in nlp(text)
                    if i.orth_ not in STOP_WORDS)
    text = ' '.join(text.split())
    return text

ml_bs_df['clean_content'] = ml_bs_df['clean_content'].apply(cleaner)

## Creating Document_term matrix using Tfidf after cleaning
tfidf = TfidfVectorizer(min_df = 10)
ml_bs_tf = tfidf.fit_transform(ml_bs_df['clean_content'])
ml_bs_tf_matrix_df = pd.DataFrame(ml_bs_tf.toarray(),
                                       index=ml_bs_df.pageid,
                                       columns=tfidf.get_feature_names())

## defining function that creates cosine similarity matrix from user supplied query and returns the top 5 similar pages
def doc_matrix(df, array, min_df, search_query, n_comp ):
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df = min_df)
    array_tf = tfidf.fit_transform(array)
    array_tf_df =  pd.DataFrame(array_tf.toarray(), index=df.pageid, columns=tfidf.get_feature_names())
    #return array_tf_df.shape
    
    query = pd.Series(search_query)
    query_tf = tfidf.transform(query)
    query_tf_df = pd.DataFrame(query_tf.toarray(), 
                                       index=query, 
                                       columns=tfidf.get_feature_names())
    query_df_tf_df = array_tf_df.append(query_tf_df)
    
    svd = TruncatedSVD(n_components= n_comp)
    component_names = ["component_"+str(i+1) for i in range(n_comp)]
    svd_matrix = svd.fit_transform(query_df_tf_df)
    svd_df = pd.DataFrame(svd_matrix, index=query_df_tf_df.index, columns=component_names)
    
    search_term_svd_vector = svd_df.loc[query_tf_df.index]
    svd_df['cosine_sim'] = cosine_similarity(svd_df, search_term_svd_vector)
    return svd_df[['cosine_sim']].sort_values('cosine_sim', ascending=False).head(5)

doc_matrix(ml_bs_df, ml_bs_df['clean_content'], 10, sys.argv[1] , 500)