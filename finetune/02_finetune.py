"""
Stage 2: Fine-tune multilingual-e5-large on Swiss legal retrieval

Usage:
    python finetune/02_finetune.py --query-mode original   --run-name scenario1-original
    python finetune/02_finetune.py --query-mode translated --run-name scenario2-translated
    python finetune/02_finetune.py --query-mode both       --run-name scenario3-both

Sources:
- ShawhinT: SentenceTransformerTrainer + MultipleNegativesRankingLoss + BatchSamplers.NO_DUPLICATES
- NVIDIA NeMo: lr=1e-5, cosine decay, warmup, weight_decay=0.01, num_epochs=3
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from finetune.config import cfg, get_model_config
from finetune.logger import setup_logger


class SwissLegalF1Evaluator(SentenceEvaluator):
    """
    Evaluator that computes Macro F1 at K (competition metric) on val.csv.
    Returns key 'swiss-legal-val_macro_f1' so metric_for_best_model can find it.
    Val is only 10 queries so F1 is noisy — use as a tiebreaker, not gospel.
    """

    def __init__(self, queries: dict, corpus: dict, relevant_docs: dict,
                 name: str = "swiss-legal-val", k_values: tuple = (5, 10, 20),
                 query_prefix: str = "", passage_prefix: str = "",
                 query_prompt_name: str = "", passage_prompt_name: str = ""):
        self.queries = queries           # {qid: query_str}
        self.corpus = corpus             # {docid: doc_str}
        self.relevant_docs = relevant_docs  # {qid: set of docids}
        self.name = name
        self.k_values = k_values
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.query_prompt_name = query_prompt_name
        self.passage_prompt_name = passage_prompt_name
        self.primary_metric = f"{name}_macro_f1"  # required by ST ≥3.0

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        q_ids = list(self.queries.keys())
        c_ids = list(self.corpus.keys())

        if self.query_prompt_name:
            q_embs = model.encode(
                list(self.queries.values()), prompt_name=self.query_prompt_name,
                normalize_embeddings=True, show_progress_bar=False, batch_size=256,
            )
        else:
            q_embs = model.encode(
                [self.query_prefix + v for v in self.queries.values()],
                normalize_embeddings=True, show_progress_bar=False, batch_size=256,
            )

        if self.passage_prompt_name:
            c_embs = model.encode(
                list(self.corpus.values()), prompt_name=self.passage_prompt_name,
                normalize_embeddings=True, show_progress_bar=False, batch_size=256,
            )
        else:
            c_embs = model.encode(
                [self.passage_prefix + v for v in self.corpus.values()],
                normalize_embeddings=True, show_progress_bar=False, batch_size=256,
            )

        scores = q_embs @ c_embs.T  # (n_queries, n_corpus)
        metrics = {}

        for k in self.k_values:
            top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
            all_f1 = []
            for i, qid in enumerate(q_ids):
                pred_set = {c_ids[j] for j in top_idx[i]}
                gold_set = self.relevant_docs.get(qid, set())
                if not pred_set and not gold_set:
                    all_f1.append(1.0)
                elif not pred_set or not gold_set:
                    all_f1.append(0.0)
                else:
                    tp = len(pred_set & gold_set)
                    p = tp / len(pred_set)
                    r = tp / len(gold_set)
                    all_f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            metrics[f"{self.name}_macro_f1@{k}"] = float(np.mean(all_f1))

        metrics[self.primary_metric] = metrics[f"{self.name}_macro_f1@10"]
        return metrics


def build_val_evaluator(val_csv: str, laws_csv: str, court_csv: str,
                        query_prefix: str, passage_prefix: str,
                        query_prompt_name: str = "",
                        passage_prompt_name: str = "") -> SwissLegalF1Evaluator:
    """
    Build SwissLegalF1Evaluator from val.csv + corpus.
    Val queries are English — matches the test distribution.
    """
    val_df = pd.read_csv(val_csv)

    # Load corpus
    parts = []
    for path in [laws_csv, court_csv]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            parts.append(df[["citation", "text"]].dropna())
    corpus_df = pd.concat(parts, ignore_index=True)
    citation_to_text = dict(zip(corpus_df["citation"], corpus_df["text"]))

    # Build evaluator inputs
    queries = {}      # qid -> query string (raw, prefix applied inside evaluator)
    corpus = {}       # docid -> doc string (raw)
    relevant_docs = {}  # qid -> set of relevant docids

    for _, row in val_df.iterrows():
        qid = str(row["query_id"])
        queries[qid] = str(row["query"])
        if pd.isna(row.get("gold_citations", None)):
            continue
        citations = [c.strip() for c in str(row["gold_citations"]).split(";")]
        relevant_docs[qid] = set()
        for cit in citations:
            if cit in citation_to_text:
                corpus[cit] = citation_to_text[cit]
                relevant_docs[qid].add(cit)

    # Use all laws_de docs as corpus — 5000 distractors was too small (gave inflated F1)
    # and didn't reflect real inference difficulty (176k docs). This aligns training
    # evaluation with actual inference conditions so best-checkpoint selection is meaningful.
    for cit, text in citation_to_text.items():
        if cit not in corpus:
            corpus[cit] = text

    print(f"  Val evaluator: {len(queries)} queries, {len(corpus):,} corpus docs")

    return SwissLegalF1Evaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="swiss-legal-val",
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        query_prompt_name=query_prompt_name,
        passage_prompt_name=passage_prompt_name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mode", choices=["original", "translated", "both"],
                        default="both")
    parser.add_argument(
        "--model", choices=["e5-large", "gemma"], default="e5-large",
        help="Embedding model to fine-tune.",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for model output directory. Defaults to '{model}-{query_mode}'")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-device batch size. Default 4 for P100 16GB.")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable fp16 (use if GPU doesn't support it)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Save all output to this file (default: finetune/logs/{run_name}.txt)")
    args = parser.parse_args()

    query_mode = args.query_mode
    model_type = args.model
    mcfg = get_model_config(model_type)
    run_name = args.run_name or f"{model_type}-{query_mode}"
    output_dir = Path(cfg.train.models_root) / run_name
    data_dir = Path(cfg.data.prepared_data_root) / model_type / query_mode
    log_file = args.log_file or f"finetune/logs/{run_name}.txt"
    setup_logger(log_file)

    num_epochs = args.epochs or cfg.train.num_epochs
    batch_size = args.batch_size or cfg.train.batch_size

    print(f"\n{'='*60}")
    print(f"  Fine-tuning — mode: {query_mode}  model: {model_type}  run: {run_name}")
    print(f"  Base model: {mcfg.base_model}")
    print(f"  Data:   {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {num_epochs}  Batch: {batch_size}  LR: {cfg.train.learning_rate}")
    print(f"{'='*60}\n")

    if not data_dir.exists():
        print(f"ERROR: Prepared data not found at {data_dir}")
        print(f"Run first: python finetune/01_prepare_data.py --query-mode {query_mode}")
        sys.exit(1)

    # --- Load dataset ---
    print("[1/4] Loading prepared dataset...")
    ds = load_from_disk(str(data_dir))
    print(f"  Train: {len(ds['train']):,}  Eval: {len(ds['eval']):,}")
    print(f"  Columns: {ds['train'].column_names}")

    # Use hard negatives if present — MNRL treats the "negative" column as an
    # explicit hard negative per triplet, giving stronger gradient signal.
    train_cols = ["anchor", "positive"]
    if "negative" in ds["train"].column_names:
        train_cols.append("negative")
        print(f"  Hard negatives column found — using triplet training")
    else:
        print(f"  No hard negatives column — using in-batch negatives only")
    train_dataset = ds["train"].select_columns(train_cols)

    # --- Load model (ShawhinT approach) ---
    print(f"\n[2/4] Loading base model: {mcfg.base_model}")
    model = SentenceTransformer(mcfg.base_model)

    # Reduce max sequence length to save VRAM
    # Legal texts are long but 256 tokens covers most article snippets
    # Data analysis shows: 93.9% of corpus docs < 128 tokens, mean=57
    # 128 covers almost all docs with 16x less memory than 512
    model.max_seq_length = 128
    print(f"  max_seq_length set to {model.max_seq_length} (93.9% docs fit fully)")

    # --- Loss (ShawhinT: MNRL) ---
    # MultipleNegativesRankingLoss: treats all other positives in batch as negatives.
    # Works well because BatchSamplers.NO_DUPLICATES prevents same anchor appearing twice.
    loss = MultipleNegativesRankingLoss(model)

    # --- Detect GPU capability for fp16 ---
    import torch
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    if use_fp16:
        try:
            _ = torch.zeros(1, dtype=torch.float16).cuda()
            print("  fp16 enabled — halves VRAM usage")
        except Exception:
            use_fp16 = False
            print("  fp16 not available, using fp32")

    # Gradient checkpointing — recompute activations during backward instead of storing
    # Saves ~60% VRAM, ~20% slower. Critical for large models on P100.
    model[0].auto_model.gradient_checkpointing_enable()
    print("  gradient checkpointing enabled — saves ~60% activation memory")

    # --- Val evaluator ---
    print("\n[3/4] Building val evaluator (English queries → German docs)...")
    # For prompt-name models (gemma), look up the actual prompt text from the model
    q_prefix = mcfg.query_prefix
    p_prefix = mcfg.passage_prefix
    if mcfg.query_prompt_name and mcfg.query_prompt_name in model.prompts:
        q_prefix = model.prompts[mcfg.query_prompt_name]
    if mcfg.passage_prompt_name and mcfg.passage_prompt_name in model.prompts:
        p_prefix = model.prompts[mcfg.passage_prompt_name]

    evaluator = build_val_evaluator(
        val_csv=cfg.data.val_csv,
        laws_csv=cfg.data.laws_csv,
        court_csv=cfg.data.court_csv,
        query_prefix=q_prefix,
        passage_prefix=p_prefix,
    )

    # --- Effective batch size via gradient accumulation ---
    # MNRL benefits from large effective batch (more in-batch negatives)
    # But actual batch per device must be small to fit in VRAM
    # effective_batch = batch_size × grad_accum_steps
    grad_accum = max(1, 16 // batch_size)  # target effective batch of 16
    print(f"\n  Batch per device: {batch_size}  ×  grad_accum: {grad_accum}"
          f"  =  effective batch: {batch_size * grad_accum}")

    # Pass prompts/prefixes to trainer so they are applied consistently during training.
    # Without this, e5-large trains on raw text but evaluates/infers with "query: "/"passage: "
    # prefixes — causing representation mismatch and performance regression.
    if mcfg.query_prompt_name:
        # Gemma-style: prompt names
        train_prompts = {
            "anchor": model.prompts.get(mcfg.query_prompt_name, ""),
            "positive": model.prompts.get(mcfg.passage_prompt_name, ""),
        }
        if "negative" in train_cols:
            train_prompts["negative"] = model.prompts.get(mcfg.passage_prompt_name, "")
    elif mcfg.query_prefix:
        # e5-style: string prefixes
        train_prompts = {
            "anchor": mcfg.query_prefix,
            "positive": mcfg.passage_prefix,
        }
        if "negative" in train_cols:
            train_prompts["negative"] = mcfg.passage_prefix
    else:
        train_prompts = None
    print(f"  Train prompts: {train_prompts}")

    # --- Training args (ShawhinT structure + NVIDIA hyperparams) ---
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.train.learning_rate,          # NVIDIA: 1e-5
        warmup_ratio=cfg.train.warmup_ratio,             # ShawhinT: 0.1
        weight_decay=cfg.train.weight_decay,             # NVIDIA: 0.01
        lr_scheduler_type=cfg.train.lr_scheduler_type,  # NVIDIA: cosine
        batch_sampler=BatchSamplers.NO_DUPLICATES,       # ShawhinT: required for MNRL
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        logging_steps=cfg.train.logging_steps,
        save_strategy="steps",
        save_steps=cfg.train.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="swiss-legal-val_macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=False,       # set True on A100/H100 only
        dataloader_pin_memory=False,
        dataloader_drop_last=True,    # avoid DDP hang on uneven last batch
        report_to="none",
        save_only_model=True,
        save_total_limit=2,           # keep only best + latest checkpoint to save disk
        prompts=train_prompts,        # None for e5 (uses raw text); prompt dict for gemma
    )

    # --- Train (ShawhinT: SentenceTransformerTrainer) ---
    print("\n[4/4] Training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=ds["eval"].select_columns(train_cols),
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    model.save(str(final_path))
    print(f"\nModel saved to {final_path}")

    # Quick eval on val
    print("\nFinal evaluation on val.csv:")
    results = evaluator(model)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
