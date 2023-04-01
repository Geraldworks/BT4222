from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
from itertools import compress

def get_text(html_text):
    soup = BeautifulSoup(html_text, features="html.parser")
    return soup.get_text()

def get_tokenized_text(txt):
    return nltk.word_tokenize(txt)

def get_nouns_and_adjs(tokenized_text):
    noun_and_adj_tags = [
        "NN", "NNS", "NNP", "NNPS",
        "JJ", "JJR", "JJS"]
    word_tags = nltk.pos_tag(tokenized_text)
    return [word_tag[1] in noun_and_adj_tags for word_tag in word_tags]

def lemmatize(word):
    return WordNetLemmatizer().lemmatize(word)

def get_top_lemmatized_noun_adj(dataset, num_stems):
    # Get stems by category
    stems_by_cat = {}
    for index, data in dataset.iterrows():
        category = data[0]
        is_noun_or_adj = data[3]
        lemmatized_tokens = data[4]
        lemmatized_nouns_and_adj = list(compress(lemmatized_tokens, is_noun_or_adj))
        if category not in stems_by_cat:
            stems_by_cat[category] = Counter()
        stems_by_cat[category].update(lemmatized_nouns_and_adj)

    # Filter to top num_stems per category
    top_stems_by_cat = {}
    for category, stems in stems_by_cat.items():
        top_stems_by_cat[category] = list(map(lambda x: x[0], stems.most_common(num_stems)))
    
    # Get bag of unique stems among top stems for all categories
    top_stems = set()
    for stems in top_stems_by_cat.values():
        for stem in stems:
            top_stems.add(stem)
    top_stems = list(top_stems)
    return top_stems

def get_tfidf(lemmatized_tokens, top_stems):
    vectorizer = TfidfVectorizer(vocabulary=top_stems)
    vectors = vectorizer.fit_transform(lemmatized_tokens.map(lambda x: " ".join(x)))
    return vectors.toarray()