import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(pipeline, X_train, y_train, X_test, y_test, target_names=None, verbose=True):
    
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    if verbose:
        print(f"Accuracy: {acc:.4f} | Training time: {elapsed:.2f}s")
        if target_names:
            print(classification_report(y_test, pred, target_names=target_names))
    return acc, pred, elapsed

def plot_confusion_matrix(y_true, y_pred, target_names, title="Confusion Matrix"):
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()