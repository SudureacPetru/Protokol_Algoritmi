from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .classifiers import get_classifier

def build_pipeline(classifier_name='LinearSVC', ngram_range=(1,2), max_features=10000):
    
    clf = get_classifier(classifier_name)
    steps = [
        ('tfidf', TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', clf)
    ]
    return Pipeline(steps)