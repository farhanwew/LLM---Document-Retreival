"""
Retrieval pipeline: MILCO sparse encoding → sparse dot product → adaptive cutoff.

Modes:
  build_index  — encode corpus and save sparse index (run once)
  tune_k       — try threshold settings on val.csv
  baseline     — tune then generate submission
  submit       — generate submission with given --k
"""
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import config
from embedder import Embedder
from indexer import SparseIndexer
from data_utils import load_val, load_test, parse_citations
from evaluate import macro_f1


class Retriever:
    def __init__(self, model_path: str = config.BASE_MODEL):
        self.embedder = Embedder(model_path)
        self.indexer = SparseIndexer()

    def build_index(self):
        """Encode corpus and build sparse index. Run once offline."""
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
        query_sparse = self.embedder.encode_queries(queries)
        all_citations, all_scores = self.indexer.search(query_sparse, top_k=top_k)

        if not adaptive:
            return [cits[:config.FINAL_TOP_K] for cits in all_citations]

        return [
            _adaptive_cutoff(cits, scrs, gap_fraction)
            for cits, scrs in zip(all_citations, all_scores)
        ]

    def tune_on_val(self, gap_values: list[float] = [0.05, 0.10, 0.15, 0.20, 0.25]):
        """Try adaptive gap fractions + fixed top-k on val.csv."""
        val = load_val()
        queries = val["query"].tolist()
        gold_list = [parse_citations(g) for g in val["gold_citations"]]

        # Encode once, reuse
        query_sparse = self.embedder.encode_queries(queries)
        all_citations, all_scores = self.indexer.search(
            query_sparse, top_k=config.RETRIEVAL_TOP_K
        )

        print(f"\n{'setting':>12} | {'Macro F1':>10}")
        print("-" * 28)

        best_setting, best_f1 = None, 0.0

        # Fixed top-k
        for k in [5, 10, 15, 20]:
            preds = [cits[:k] for cits in all_citations]
            f1 = macro_f1(gold_list, preds)
            print(f"{'top-'+str(k):>12} | {f1:>10.4f}")
            if f1 > best_f1:
                best_f1, best_setting = f1, ("fixed", k)

        # Adaptive gap
        for gap in gap_values:
            preds = [
                _adaptive_cutoff(cits, scrs, gap)
                for cits, scrs in zip(all_citations, all_scores)
            ]
            f1 = macro_f1(gold_list, preds)
            print(f"{'gap='+str(gap):>12} | {f1:>10.4f}")
            if f1 > best_f1:
                best_f1, best_setting = f1, ("adaptive", gap)

        print(f"\n→ Best: {best_setting} (F1 = {best_f1:.4f})")
        return best_setting

    def generate_submission(
        self,
        top_k: int = config.RETRIEVAL_TOP_K,
        adaptive: bool = True,
        gap_fraction: float = config.ADAPTIVE_GAP_FRACTION,
        output_path: Path = config.SUBMISSION_PATH,
    ):
        test = load_test()
        queries = test["query"].tolist()
        print(f"[submit] retrieving for {len(queries)} queries ...")
        preds = self.retrieve(queries, top_k=top_k, adaptive=adaptive, gap_fraction=gap_fraction)

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
    """Return citations up to the largest score gap > gap_fraction * top_score."""
    if len(citations) <= 1:
        return citations

    top_score = float(scores[0])
    threshold = gap_fraction * top_score
    best_cut, best_gap = len(citations), 0.0

    for i in range(len(citations) - 1):
        gap = float(scores[i]) - float(scores[i + 1])
        if gap > threshold and gap > best_gap:
            best_gap, best_cut = gap, i + 1

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
    parser.add_argument("--gap", type=float, default=config.ADAPTIVE_GAP_FRACTION)
    args = parser.parse_args()

    retriever = Retriever(model_path=args.model)

    if args.mode == "build_index":
        retriever.build_index()

    elif args.mode == "tune_k":
        retriever.load_index()
        retriever.tune_on_val()

    elif args.mode == "baseline":
        retriever.load_index()
        best = retriever.tune_on_val()
        kind, val = best
        if kind == "fixed":
            retriever.generate_submission(top_k=val, adaptive=False)
        else:
            retriever.generate_submission(adaptive=True, gap_fraction=val)

    elif args.mode == "submit":
        retriever.load_index()
        retriever.generate_submission(top_k=args.k, gap_fraction=args.gap)
