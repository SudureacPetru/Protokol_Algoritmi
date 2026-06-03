from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_classifier(name):
    
    if name == 'NaiveBayes':
        return MultinomialNB()
    elif name == 'LinearSVC':
        return LinearSVC(max_iter=2000)
    elif name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, solver='saga')
    elif name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Clasificator necunoscut: {name}")