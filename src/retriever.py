"""
Full retrieval pipeline: embed queries → FAISS search → adaptive threshold.

Modes (python src/retriever.py --mode <mode>):
  build_index  — embed corpus and build FAISS index (run once)
  tune_k       — try fixed k values on val.csv
  baseline     — tune k on val then generate submission
  submit       — generate submission with given --k
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import config
from embedder import Embedder
from indexer import FaissIndexer
from data_utils import load_val, load_test, parse_citations
from evaluate import macro_f1


class Retriever:
    def __init__(self, model_path: str = config.BASE_MODEL):
        self.embedder = Embedder(model_path)
        self.indexer = FaissIndexer()

    def build_index(self):
        """Embed corpus and build FAISS index. Run once offline."""
        from data_utils import load_corpus
        corpus = load_corpus()
        self.indexer.build(corpus, self.embedder)
        self.indexer.save()

    def load_index(self):
        self.indexer.load()

    def retrieve(
        self,
        queries: list[str],
        top_k: int = config.RETRIEVAL_TOP_K,
        adaptive: bool = True,
        gap_fraction: float = config.ADAPTIVE_GAP_FRACTION,
    ) -> list[list[str]]:
        """
        Retrieve citations for each query.

        If adaptive=True, uses a score-gap heuristic to decide how many
        citations to return per query (better Macro F1 than fixed k).
        If adaptive=False, returns exactly top_k citations.
        """
        query_embs = self.embedder.encode_queries(queries)
        scores, indices = self.indexer.search(query_embs, top_k=top_k)
        all_citations = self.indexer.get_citations(indices)

        if not adaptive:
            return all_citations

        return [
            _adaptive_cutoff(cits, scrs, gap_fraction)
            for cits, scrs in zip(all_citations, scores)
        ]

    def tune_k_on_val(self, k_values: list[int] = [5, 10, 15, 20, 30]):
        """Try fixed k values on val.csv and report Macro F1."""
        val = load_val()
        queries = val["query"].tolist()
        gold_list = [parse_citations(g) for g in val["gold_citations"]]

        # Embed once, reuse for all k values
        query_embs = self.embedder.encode_queries(queries)
        scores, indices = self.indexer.search(query_embs, top_k=max(k_values))
        citations_list = self.indexer.get_citations(indices)

        print(f"\n{'k':>5} | {'Macro F1':>10}")
        print("-" * 20)
        best_k, best_f1 = k_values[0], 0.0
        for k in k_values:
            preds = [cits[:k] for cits in citations_list]
            f1 = macro_f1(gold_list, preds)
            print(f"{k:>5} | {f1:>10.4f}")
            if f1 > best_f1:
                best_f1, best_k = f1, k

        # Also try adaptive threshold
        adaptive_preds = [
            _adaptive_cutoff(cits, scrs, config.ADAPTIVE_GAP_FRACTION)
            for cits, scrs in zip(citations_list, scores)
        ]
        adaptive_f1 = macro_f1(gold_list, adaptive_preds)
        print(f"{'auto':>5} | {adaptive_f1:>10.4f}  (adaptive gap={config.ADAPTIVE_GAP_FRACTION})")

        if adaptive_f1 > best_f1:
            print(f"\n→ Best = adaptive (F1 = {adaptive_f1:.4f})")
        else:
            print(f"\n→ Best k = {best_k} (F1 = {best_f1:.4f})")
        return best_k

    def generate_submission(
        self,
        top_k: int = config.RETRIEVAL_TOP_K,
        adaptive: bool = True,
        output_path: Path = config.SUBMISSION_PATH,
    ):
        test = load_test()
        queries = test["query"].tolist()
        print(f"[submit] retrieving for {len(queries)} test queries ...")
        preds = self.retrieve(queries, top_k=top_k, adaptive=adaptive)

        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        rows = [
            {"query_id": qid, "predicted_citations": ";".join(citations)}
            for qid, citations in zip(test["query_id"], preds)
        ]
        sub_df = pd.DataFrame(rows)
        sub_df.to_csv(output_path, index=False)
        print(f"[submit] saved → {output_path}")
        return sub_df


def _adaptive_cutoff(
    citations: list[str],
    scores: np.ndarray,
    gap_fraction: float,
) -> list[str]:
    """
    Return citations up to the largest score drop.

    Finds the biggest absolute gap between consecutive scores that also
    exceeds gap_fraction * top_score. Falls back to all citations if
    no significant gap is found. Always returns at least 1 citation.
    """
    if len(citations) <= 1:
        return citations

    top_score = float(scores[0])
    threshold = gap_fraction * top_score
    best_cut = len(citations)  # default: return all
    best_gap = 0.0

    for i in range(len(citations) - 1):
        gap = float(scores[i]) - float(scores[i + 1])
        if gap > threshold and gap > best_gap:
            best_gap = gap
            best_cut = i + 1

    return citations[:best_cut]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["build_index", "tune_k", "baseline", "submit"],
        default="baseline",
    )
    parser.add_argument("--model", default=config.BASE_MODEL)
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_TOP_K)
    args = parser.parse_args()

    retriever = Retriever(model_path=args.model)

    if args.mode == "build_index":
        retriever.build_index()

    elif args.mode == "tune_k":
        retriever.load_index()
        retriever.tune_k_on_val()

    elif args.mode == "baseline":
        retriever.load_index()
        best_k = retriever.tune_k_on_val()
        retriever.generate_submission(top_k=best_k)

    elif args.mode == "submit":
        retriever.load_index()
        retriever.generate_submission(top_k=args.k)
