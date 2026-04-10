"""
Hybrid retrieval: BM25 first pass → MILCO rerank → adaptive cutoff.

Modes:
  build_index  — build BM25 index from corpus (run once, CPU only)
  tune_k       — find best threshold settings on val.csv
  baseline     — tune then generate submission
  submit       — generate submission
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import config
from bm25_index import BM25Index
from embedder import Embedder
from data_utils import load_val, load_test, parse_citations
from evaluate import macro_f1


class Retriever:
    def __init__(self, model_path: str = config.BASE_MODEL):
        self.bm25 = BM25Index()
        self.embedder = Embedder(model_path)

    def build_index(self):
        """Build BM25 index from corpus. CPU only, run once."""
        from data_utils import load_corpus
        corpus = load_corpus()
        self.bm25.build(corpus)
        self.bm25.save()

    def load_index(self):
        self.bm25.load()

    def retrieve(
        self,
        queries: list[str],
        bm25_top_k: int = config.BM25_TOP_K,
        rerank_top_k: int = config.RERANK_TOP_K,
        adaptive: bool = True,
        gap_fraction: float = config.ADAPTIVE_GAP_FRACTION,
    ) -> list[list[str]]:
        """
        For each query:
          1. BM25 → top bm25_top_k candidates
          2. MILCO rerank candidates
          3. Adaptive cutoff → final citations
        """
        results = []
        for query in tqdm(queries, desc="retrieving"):
            candidates = self.bm25.search(query, top_k=bm25_top_k)
            if not candidates:
                results.append([])
                continue

            texts = [c["text"] for c in candidates]
            citations = [c["citation"] for c in candidates]

            # MILCO rerank
            q_sparse = self.embedder.encode_queries([query])
            d_sparse = self.embedder.encode_documents(texts)
            scores = (q_sparse @ d_sparse.T).toarray()[0].astype(float)

            order = np.argsort(scores)[::-1]
            ranked_citations = [citations[i] for i in order]
            ranked_scores = scores[order]

            if adaptive:
                predicted = _adaptive_cutoff(ranked_citations, ranked_scores, gap_fraction)
            else:
                predicted = ranked_citations[:rerank_top_k]

            results.append(predicted)

        return results

    def tune_on_val(self):
        """Try gap fractions + fixed-k on val.csv and report Macro F1."""
        val = load_val()
        queries = val["query"].tolist()
        gold_list = [parse_citations(g) for g in val["gold_citations"]]

        # BM25 + MILCO once, then try different cutoffs
        all_citations, all_scores = [], []
        for query in tqdm(queries, desc="tuning"):
            candidates = self.bm25.search(query, top_k=config.BM25_TOP_K)
            texts = [c["text"] for c in candidates]
            citations = [c["citation"] for c in candidates]

            q_sparse = self.embedder.encode_queries([query])
            d_sparse = self.embedder.encode_documents(texts)
            scores = (q_sparse @ d_sparse.T).toarray()[0].astype(float)

            order = np.argsort(scores)[::-1]
            all_citations.append([citations[i] for i in order])
            all_scores.append(scores[order])

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
        adaptive: bool = True,
        gap_fraction: float = config.ADAPTIVE_GAP_FRACTION,
        rerank_top_k: int = config.RERANK_TOP_K,
        output_path: Path = config.SUBMISSION_PATH,
    ):
        test = load_test()
        queries = test["query"].tolist()
        print(f"[submit] retrieving for {len(queries)} queries ...")
        preds = self.retrieve(
            queries, adaptive=adaptive, gap_fraction=gap_fraction, rerank_top_k=rerank_top_k
        )

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
            retriever.generate_submission(adaptive=False, rerank_top_k=val)
        else:
            retriever.generate_submission(adaptive=True, gap_fraction=val)

    elif args.mode == "submit":
        retriever.load_index()
        retriever.generate_submission(gap_fraction=args.gap)
