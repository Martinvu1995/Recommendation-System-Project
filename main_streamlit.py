# import libraries

import streamlit as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize , pos_tag, sent_tokenize
import warnings
from gensim import corpora , models , similarities
import jieba
import re

# create dataframes

# 1.Read data
data_product_original = pd.read_excel('data_news_3.xlsx')  
data_product_original.tail(2)

# drop nan
data_product_original = data_product_original.dropna()

# GUI

st.title('Data Science Project')
st.write('## Recommendation System Project')
st.write('### i used Gensim algo to solve this problem.')
st.write('#### Gensim is a Python library for topic modelling, document indexing and similarity \
      retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.')

st.image('Gensim.png', width = 500)

# Upload file

uploaded_file = st.file_uploader('Choose a file', type = ['xlsx, csv'])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file )
    df.to_excel('data_news_3_new.xlsx', index = False)

# 2.Data Pre-processing


data_product = data_product_original['description'].to_frame()
# data_product.tail(2)
# change name column

data_product.columns = [ 'description']

# check df
# add 'link' column to data_product
data_product['link'] = data_product_original['link']
data_product.head(2)

# check nan
data_product.isnull().sum()

# drop nan
data_product = data_product.dropna()

# 3. Text Pre-processing

# word tokenize

data_product['description_wt'] = data_product['description'].apply(lambda x: word_tokenize(x, format="text"))

# check df
data_product.head(2)

# pos tag

product_pos_tag = data_product['description'].apply(lambda x: pos_tag(x, format="text"))


# sent_tokenize 
product_sent = data_product['description'].apply(lambda x: sent_tokenize(x))


# 4. GENSIM

# tokenize(split) the sentences into words
products_gem = [[text for text in x.split()] for x in data_product.description_wt]

# using regular expression to remove special characters
import re

with open('vietnamese-stopwords.txt' , 'r', encoding='utf8') as stop_words:
    stop_words = stop_words.read().split('\n')

# remove some special characters
def ham_xu_ly(stop_words , products_gem):
    products_gem_re = [[re.sub('[0-9]+' , '', e) for e in text] for text in products_gem] # so
    products_gem_re = [[t.lower() for t in text if not t in['', ' ', ',', '.', '...', '-',':',';','?', '%','(', ')','+', '/', 'g','ml']]for text in products_gem_re] # ky tu dac biet
    products_gem_re = [[t for t in text if not t in stop_words] for text in products_gem_re]  # stopword
    return products_gem_re

products_gem_re = ham_xu_ly(stop_words , products_gem)

# using gensim to create dictionary
dictionary = corpora.Dictionary(products_gem_re)

# list of words in the dictionary
# dictionary.token2id

# feature_cnt
feature_cnt = len(dictionary.token2id.keys())

# get corpus
corpus = [dictionary.doc2bow(text) for text in products_gem_re]

# use tfidf model to transform the corpus
tfidf = models.TfidfModel(corpus)
#  sparse matrix
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)

# select the second product as the query

product_selection = data_product.iloc[2]

# convert to dataframe

product_selection = pd.DataFrame(product_selection).T

# get the product name
name_description_pre = product_selection['description_wt'].to_string(index=False)

# view_product
view_product = name_description_pre.lower().split()

# convert search product to vector
kw_vector = dictionary.doc2bow(view_product)

# similarity calculation
sims = index[tfidf[kw_vector]]

# sort the similarity
sims = sorted(enumerate(sims), key=lambda item: -item[1])

# get the most similar products
# sims[:5]

# get name of the most similar products

def get_name_product(sims):
    name_product = []
    for i in sims:
        name_product.append(data_product['description'].iloc[i[0]])
    return name_product

get_name_product(sims[:3])

################GUI####################

menu = ['Recommendation System', 'Build Model', 'Prediction'] 

choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Recommendation System':
    st.subheader('Recommendation System')
    st.write('''
    ###### This is a recommendation system project. The system will recommend the most similar products based on the product you choose.''')
    st.image('image_1.jpeg', width = 700)

if choice == 'Build Model':
    st.subheader('Build Model')
    st.write('##### 1.Some data')
    st.dataframe(data_product.head(5))

if choice == 'Prediction':
    st.subheader('Prediction')

    # input 'link' columns in data_product_original dataframe to get the most similar products

    link = st.text_input('Enter link of product')
    if link:
        product_selection = data_product[data_product['link'] == link]
        name_description_pre = product_selection['description_wt'].to_string(index=False)
        view_product = name_description_pre.lower().split()
        kw_vector = dictionary.doc2bow(view_product)
        sims = index[tfidf[kw_vector]]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        # return a dataframe of the most similar products
        prediction = data_product[data_product['description'].isin(get_name_product(sims[:3]))]
        
        # st.write('##### fisrt product is the product you choose')
        st.write(prediction)
        st.write('##### 2.The most similar products')
        # st.write(get_name_product(sims[:3]))

    # input 'description' columns in data_product_original dataframe to get the most similar products

    description = st.text_input('Enter description of product')
    if description:
        product_selection = data_product[data_product['description'] == description]
        name_description_pre = product_selection['description_wt'].to_string(index=False)
        view_product = name_description_pre.lower().split()
        kw_vector = dictionary.doc2bow(view_product)
        sims = index[tfidf[kw_vector]]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        st.write('##### 2.The most similar products')
        
        st.write('##### fisrt product is the product you choose')
        prediction = data_product[data_product['description'].isin(get_name_product(sims[:3]))]
        st.write(prediction)
    