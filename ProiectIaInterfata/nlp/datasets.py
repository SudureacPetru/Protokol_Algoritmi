from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import movie_reviews
import random

def load_20news(subset='train', categories=None):
    """Încarcă 20 Newsgroups (toate cele 20 de categorii)."""
    if categories is None:
        categories = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
            'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
            'sci.space', 'soc.religion.christian', 'talk.politics.guns',
            'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
        ]
    data = fetch_20newsgroups(subset=subset, categories=categories,
                              remove=('headers','footers','quotes'))
    return data.data, data.target, data.target_names


def load_imdb_nltk():
    """
    Încarcă dataset-ul IMDB din NLTK (2000 recenzii, clasificare binară).
    Returnează X (texte), y (0/1), target_names = ['neg', 'pos'].
    """
    # Asigură-te că ai făcut download o dată: python -m nltk.downloader movie_reviews
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        raise RuntimeError(
            "NLTK movie_reviews nu e descărcat. Rulează:\n"
            "python -m nltk.downloader movie_reviews"
        )
    
    documents = []
    labels = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(movie_reviews.raw(fileid))
            labels.append(category)
    
    # Etichete numerice: 0 = neg, 1 = pos
    target = [0 if lbl == 'neg' else 1 for lbl in labels]
    target_names = ['neg', 'pos']
    
    # Amestecăm pentru a nu avea toate negativele la început
    combined = list(zip(documents, target))
    random.seed(42)
    random.shuffle(combined)
    documents, target = zip(*combined)
    
    return list(documents), list(target), target_names