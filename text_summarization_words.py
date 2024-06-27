import streamlit as st
import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def build_similarity_matrix(sentences, tfidf_matrix):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0]
    return similarity_matrix

def textrank_summarize(text, max_words=100):
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = build_similarity_matrix(sentences, tfidf_matrix)
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    summary = []
    current_word_count = 0

    for score, sentence in ranked_sentences:
        word_count = len(word_tokenize(sentence))
        if current_word_count + word_count <= max_words:
            summary.append(sentence)
            current_word_count += word_count
        else:
            break
    
    return ' '.join(summary)

# Streamlit app
st.title("Text Summarization App")
st.write("This app summarizes text using the TextRank algorithm based on the number of words.")

# Input text area
text = st.text_area("Enter text to summarize", height=300)

# Number of words for the summary
max_words = st.slider("Number of words in summary", min_value=10, max_value=500, value=100)

# Button to generate summary
if st.button("Summarize"):
    if text:
        summary = textrank_summarize(text, max_words=max_words)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.write("Please enter text to summarize.")

# To run the app, save this script and use the command `streamlit run script_name.py`
