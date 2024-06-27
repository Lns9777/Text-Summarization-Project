import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import streamlit as st

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def build_similarity_matrix(sentences, tfidf_matrix):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    for i in range (len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
    
    return similarity_matrix

def textrank_summarize(text, top_n=5):
    # split the text into sentences
    sentences = sent_tokenize(text)

    # preprocess sentences and convert them to a TF-IDF Matix
    tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    # Build a similarity matrix
    similarity_matrix = build_similarity_matrix(sentences, tfidf_matrix)

    # Build a graph and apply PageRank algorithm
    graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(graph)
    
    # Sort the sentences based on their scores
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    summary = " ".join([sentence for score, sentence in ranked_sentences[:top_n]])
    return summary

st.title("Text Summarization App")

st.write("This app summarizes text using the TextRank Algorithm")

# Input text area
text = st.text_area('Enter text to summarize',height=300)

# Number of sentences for the summary
top_n = st.slider('Number of sentences for the summary',min_value=1,max_value=15, value=3)

# Button to generate summary
if st.button("Summarize"):
    if text :
        summary = textrank_summarize(text, top_n=top_n)
        st.write(summary)
    else:
        st.write("Please Enter text to Summarize")

