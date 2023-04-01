from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pandas as pd

def get_text(html_text):
    soup = BeautifulSoup(html_text, features="html.parser")
    return soup.get_text()

def embed(txt):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    txt = [txt]
    embeddings = model.encode(txt)
    for sentence, embedding in zip(txt, embeddings):
        return embedding