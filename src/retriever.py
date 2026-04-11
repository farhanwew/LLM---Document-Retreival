"""
Hybrid retrieval: MILCO Sparse Index Search → adaptive cutoff.
Uses Learned Sparse Retrieval (LSR) to bridge English queries to German/French/Italian docs.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config
from milco_index import MILCOIndex
from embedder import Embedder
from data_utils import load_val, load_test, parse_citations
from evaluate import macro_f1


class Retriever:
    def __init__(self, model_path: str = config.BASE_MODEL):
        self.embedder = Embedder(model_path)
        self.milco_index = MILCOIndex(self.embedder)

    def build_index(self):
        """Build MILCO index from full corpus. GPU intensive."""
        from data_utils import load_corpus
        corpus = load_corpus()
        self.milco_index.build(corpus)
        self.milco_index.save()

    def build_mini_index(self):
        """Build MILCO index only from documents in train.csv citations."""
        from data_utils import load_train, load_mini_corpus
        train = load_train()
        mini_corpus = load_mini_corpus(train)
        self.milco_index.build(mini_corpus)
        mini_path = config.MODELS_DIR / "milco_mini.npz"
        mini_meta = config.MODELS_DIR / "milco_mini_meta.pkl"
        self.milco_index.save(mini_path, mini_meta)

    def load_index(self, path: Path = config.MILCO_INDEX_PATH, meta_path: Path = config.MILCO_META_PATH):
        self.milco_index.load(path, meta_path)

    def retrieve(
        self,
        queries: list[str],
        top_k: int = config.BM25_TOP_K,
        adaptive: bool = True,
        gap_fraction: float = config.ADAPTIVE_GAP_FRACTION,
    ) -> list[list[str]]:
        """
        1. Encode query to English sparse lexical space
        2. Dot product search in MILCO index
        3. Adaptive cutoff
        """
        print(f"[retriever] encoding {len(queries)} queries ...")
        q_vectors = self.embedder.encode_queries(queries)
        
        print(f"[retriever] searching index ...")
        batch_results = self.milco_index.search_batch(q_vectors, top_k=top_k)

        results = []
        for query_res in batch_results:
            if not query_res:
                results.append([])
                continue

            citations = [c["citation"] for c in query_res]
            scores = np.array([c["score"] for c in query_res])

            if adaptive:
                predicted = _adaptive_cutoff(citations, scores, gap_fraction)
            else:
                predicted = citations[:config.RERANK_TOP_K]

            results.append(predicted)

        return results

    def tune_on_val(self):
        val = load_val()
        queries = val["query"].tolist()
        gold_list = [parse_citations(g) for g in val["gold_citations"]]

        print(f"[tune] searching for {len(queries)} validation queries ...")
        q_vectors = self.embedder.encode_queries(queries)
        batch_results = self.milco_index.search_batch(q_vectors, top_k=config.BM25_TOP_K)

        all_citations, all_scores = [], []
        for query_res in batch_results:
            all_citations.append([c["citation"] for c in query_res])
            all_scores.append(np.array([c["score"] for c in query_res]))

        print(f"\n{'setting':>12} | {'Macro F1':>10}")
        print("-" * 28)
        best_setting, best_f1 = None, 0.0

        for k in [5, 10, 15, 20]:
            preds = [cits[:k] for cits in all_citations]
            f1 = macro_f1(gold_list, preds)
            print(f"{'top-'+str(k):>12} | {f1:>10.4f}")
            if f1 > best_f1:
                best_f1, best_setting = f1, ("fixed", k)

        for gap in [0.05, 0.10, 0.15, 0.20, 0.25]:
            preds = [_adaptive_cutoff(cits, scrs, gap) for cits, scrs in zip(all_citations, all_scores)]
            f1 = macro_f1(gold_list, preds)
            print(f"{'gap='+str(gap):>12} | {f1:>10.4f}")
            if f1 > best_f1:
                best_f1, best_setting = f1, ("adaptive", gap)

        print(f"\n→ Best: {best_setting} (F1 = {best_f1:.4f})")
        return best_setting

    def generate_submission(
        self,
        adaptive: bool = True,
        gap_fraction: float = config.ADAPTIVE_GAP_FRACTION,
        rerank_top_k: int = config.RERANK_TOP_K,
        output_path: Path = config.SUBMISSION_PATH,
    ):
        test = load_test()
        queries = test["query"].tolist()
        preds = self.retrieve(queries, adaptive=adaptive, gap_fraction=gap_fraction)
        
        rows = [{"query_id": qid, "predicted_citations": ";".join(citations)} for qid, citations in zip(test["query_id"], preds)]
        sub_df = pd.DataFrame(rows)
        sub_df.to_csv(output_path, index=False)
        print(f"[submit] saved → {output_path}")
        return sub_df


def _adaptive_cutoff(citations: list[str], scores: np.ndarray, gap_fraction: float) -> list[str]:
    if len(citations) == 0: return []
    if len(citations) == 1: return citations
    
    top_score = float(scores[0])
    threshold = gap_fraction * top_score
    best_cut = 1
    best_gap = 0.0
    
    for i in range(len(citations) - 1):
        gap = float(scores[i]) - float(scores[i + 1])
        if gap > threshold and gap > best_gap:
            best_gap, best_cut = gap, i + 1
    return citations[:best_cut]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build_index", "build_mini_index", "tune_k", "submit"], default="submit")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--gap", type=float, default=config.ADAPTIVE_GAP_FRACTION)
    args = parser.parse_args()

    retriever = Retriever()

    if args.mode == "build_index":
        retriever.build_index()
    elif args.mode == "build_mini_index":
        retriever.build_mini_index()
    elif args.mode == "tune_k":
        path = config.MODELS_DIR / "milco_mini.npz" if args.mini else config.MILCO_INDEX_PATH
        meta = config.MODELS_DIR / "milco_mini_meta.pkl" if args.mini else config.MILCO_META_PATH
        retriever.load_index(path, meta)
        retriever.tune_on_val()
    elif args.mode == "submit":
        retriever.load_index()
        retriever.generate_submission(gap_fraction=args.gap)
