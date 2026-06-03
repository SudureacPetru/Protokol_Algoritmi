from tsp.benchmark import run_comparison
from nlp.benchmark import compare_classifiers, study_ngram_range, study_max_features

if __name__ == "__main__":
    print("Rulez benchmark TSP...")
    run_comparison(sizes=[5, 8, 10], seeds=[42])

    print("\n=== NLP Benchmark ===")
    
    compare_classifiers('20news', ngram_range=(1,2), max_features=10000)
    
    study_ngram_range('agnews', max_features=5000)
    
    study_max_features('agnews', ngram_range=(1,2))