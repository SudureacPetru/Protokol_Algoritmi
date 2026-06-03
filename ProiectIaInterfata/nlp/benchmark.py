import matplotlib.pyplot as plt
import pandas as pd
from .datasets import load_20news, load_ag_news
from .pipeline import build_pipeline
from .evaluate import evaluate_model, plot_confusion_matrix

def compare_classifiers(dataset_name='20news', ngram_range=(1,1), max_features=None):
    """
    Compară 4 clasificatori pe datasetul ales.
    dataset_name: '20news' sau 'agnews'
    """
    if dataset_name == '20news':
        X_train, y_train, target_names = load_20news(subset='train')
        X_test, y_test, _ = load_20news(subset='test')
    elif dataset_name == 'agnews':
        X, y, target_names = load_ag_news()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        raise ValueError("Dataset necunoscut")

    classifiers = ['NaiveBayes', 'LinearSVC', 'LogisticRegression', 'RandomForest']
    results = []
    for clf_name in classifiers:
        pipe = build_pipeline(clf_name, ngram_range=ngram_range, max_features=max_features)
        acc, pred, t = evaluate_model(pipe, X_train, y_train, X_test, y_test,
                                      target_names, verbose=False)
        results.append((clf_name, acc, t))
        
        if clf_name == 'LinearSVC':
            plot_confusion_matrix(y_test, pred, target_names,
                                  title=f"Confusion Matrix - {clf_name} on {dataset_name}")

    
    df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Time'])
    fig, ax = plt.subplots()
    ax.bar(df['Classifier'], df['Accuracy'], color='steelblue')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Classifier Comparison on {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'classifier_comparison_{dataset_name}.png')
    plt.show()
    return df

def study_ngram_range(dataset_name='20news', classifier='LinearSVC', max_features=10000):
    """Studiu al influenței ngram_range."""
    if dataset_name == '20news':
        X_train, y_train, _ = load_20news(subset='train')
        X_test, y_test, _ = load_20news(subset='test')
    else:
        X, y, _ = load_ag_news()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ngrams = [(1,1), (1,2), (2,2), (1,3)]
    accs = []
    for ng in ngrams:
        pipe = build_pipeline(classifier, ngram_range=ng, max_features=max_features)
        acc, _, _ = evaluate_model(pipe, X_train, y_train, X_test, y_test, verbose=False)
        accs.append(acc)
    plt.bar([str(ng) for ng in ngrams], accs)
    plt.ylabel('Accuracy')
    plt.xlabel('ngram_range')
    plt.title(f'Influence of ngram_range ({classifier} on {dataset_name})')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'ngram_study_{dataset_name}.png')
    plt.show()

def study_max_features(dataset_name='20news', classifier='LinearSVC', ngram_range=(1,1)):
    
    if dataset_name == '20news':
        X_train, y_train, _ = load_20news(subset='train')
        X_test, y_test, _ = load_20news(subset='test')
    else:
        X, y, _ = load_ag_news()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    vals = [100, 500, 1000, 5000, 10000, None]
    accs = []
    labels = [str(v) for v in vals]
    for mf in vals:
        pipe = build_pipeline(classifier, ngram_range=ngram_range, max_features=mf)
        acc, _, _ = evaluate_model(pipe, X_train, y_train, X_test, y_test, verbose=False)
        accs.append(acc)
    plt.bar(labels, accs)
    plt.ylabel('Accuracy')
    plt.xlabel('max_features')
    plt.title(f'Influence of max_features ({classifier} on {dataset_name})')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'max_features_study_{dataset_name}.png')
    plt.show()