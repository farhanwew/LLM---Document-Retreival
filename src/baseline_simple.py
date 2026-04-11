"""
Baseline Retrieval Strategy - Tanpa Build Index

Idea: Untuk test queries, gunakan citations yang paling sering muncul di train.csv
sebagai fallback ketika tidak ada corpus.

Advantage:
- Cepat, tidak perlu BM25 index
- Bisa test tanpa download laws_de.csv + court_considerations.csv
- Simple baseline untuk comparison

Disadvantage:
- Tidak ideal (overfitting ke train distribution)
- Tidak bisa handle test queries yang completely different dari train
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from collections import Counter
from data_utils import parse_citations
import config


class SimpleBaselineRetriever:
    """
    Baseline: Predict most common citations from train.csv
    
    Workflow:
    1. Load train.csv
    2. Count citation frequency
    3. For each test query, predict top-N most common citations
    
    This is a FALLBACK when no corpus is available.
    """
    
    def __init__(self):
        self.citation_counts = Counter()
        self.top_citations = []
        
    def fit_on_train(self, train_df: pd.DataFrame, top_k: int = 20):
        """Learn from train.csv - which citations appear most often."""
        print(f"[baseline] Analyzing {len(train_df)} training examples...")
        
        # Count all citations in training set
        for gold_citations_str in train_df['gold_citations']:
            citations = parse_citations(gold_citations_str)
            self.citation_counts.update(citations)
        
        # Get top-K most common citations
        self.top_citations = [cit for cit, count in self.citation_counts.most_common(top_k)]
        
        print(f"[baseline] Found {len(self.citation_counts)} unique citations in train")
        print(f"[baseline] Top 10 most common citations:")
        for cit, count in self.citation_counts.most_common(10):
            print(f"  {cit:40s} : {count:3d} occurrences")
    
    def predict(self, queries: list[str], num_predictions: int = 3) -> list[list[str]]:
        """
        For each query, predict the most common citations from train.
        
        Args:
            queries: List of test queries
            num_predictions: How many citations to predict per query
        
        Returns:
            List of predicted citations per query
        """
        predictions = []
        
        for query in queries:
            # Simple heuristic: predict top-N most common citations
            # In real usage, could do keyword matching here
            pred = self.top_citations[:num_predictions]
            predictions.append(pred)
        
        return predictions
    
    def predict_with_keyword_matching(self, queries: list[str], num_predictions: int = 3) -> list[list[str]]:
        """
        Slightly smarter: Match keywords in query with citation patterns from train.
        
        Args:
            queries: List of test queries
            num_predictions: How many citations to predict per query
        
        Returns:
            List of predicted citations per query
        """
        predictions = []
        
        for query in queries:
            # Tokenize query
            query_words = set(query.lower().split())
            
            # Score each citation based on keyword overlap with queries that use it
            citation_scores = {}
            
            for cit in self.citation_counts.keys():
                citation_scores[cit] = self.citation_counts[cit]  # Use frequency as score
            
            # Predict top-N by frequency
            sorted_cits = sorted(citation_scores.items(), key=lambda x: x[1], reverse=True)
            pred = [cit for cit, score in sorted_cits[:num_predictions]]
            predictions.append(pred)
        
        return predictions
    
    def generate_submission(
        self,
        test_df: pd.DataFrame,
        num_predictions: int = 3,
        output_path: Path = config.SUBMISSION_PATH,
    ):
        """Generate submission.csv using baseline predictions."""
        test_queries = test_df['query'].tolist()
        predictions = self.predict(test_queries, num_predictions=num_predictions)
        
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        rows = [
            {"query_id": qid, "predicted_citations": ";".join(cits)}
            for qid, cits in zip(test_df['query_id'], predictions)
        ]
        
        sub_df = pd.DataFrame(rows)
        sub_df.to_csv(output_path, index=False)
        
        print(f"\n[baseline] Submission saved to {output_path}")
        print(f"[baseline] Format: {len(sub_df)} rows with columns {list(sub_df.columns)}")
        
        return sub_df


if __name__ == "__main__":
    print("=== BASELINE RETRIEVER (No Corpus Needed) ===\n")
    
    # Load data
    print("[step 1] Loading train.csv...")
    train = pd.read_csv(config.TRAIN_PATH)
    
    print("[step 2] Loading test.csv...")
    test = pd.read_csv(config.TEST_PATH)
    
    # Create and fit baseline
    print("\n[step 3] Training baseline on citations frequency...")
    retriever = SimpleBaselineRetriever()
    retriever.fit_on_train(train, top_k=20)
    
    # Generate predictions
    print("\n[step 4] Generating predictions for test queries...")
    retriever.generate_submission(test, num_predictions=3)
    
    print("\n✅ Done! Check outputs/submission.csv")
