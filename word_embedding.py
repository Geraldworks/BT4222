from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

def get_text(html_text):
    soup = BeautifulSoup(html_text, features="html.parser")
    txt = soup.get_text()
    txt = txt.split()
    stop_words = stopwords.words('english')
    txt = [x for x in txt if x not in stop_words]
    txt = " ".join(txt)
    return txt

def embed(txt):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    txt = [txt]
    embeddings = model.encode(txt)
    for sentence, embedding in zip(txt, embeddings):
        return embedding