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

import pandas as pd
from datasets import DatasetDict, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from finetune.config import cfg


def build_val_evaluator(val_csv: str, laws_csv: str, court_csv: str,
                        query_prefix: str, passage_prefix: str) -> InformationRetrievalEvaluator:
    """
    Build InformationRetrievalEvaluator from val.csv + corpus.
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
    queries = {}      # qid -> query string
    corpus = {}       # docid -> doc string
    relevant_docs = {}  # qid -> set of relevant docids

    for _, row in val_df.iterrows():
        qid = str(row["query_id"])
        queries[qid] = query_prefix + str(row["query"])
        if pd.isna(row.get("gold_citations", None)):
            continue
        citations = [c.strip() for c in str(row["gold_citations"]).split(";")]
        relevant_docs[qid] = set()
        for cit in citations:
            if cit in citation_to_text:
                docid = cit
                corpus[docid] = passage_prefix + citation_to_text[cit]
                relevant_docs[qid].add(docid)

    # Add some distractor docs from corpus (not in val positives) for realistic evaluation
    all_cits = list(citation_to_text.keys())
    pos_cits = set(corpus.keys())
    distractors = [c for c in all_cits if c not in pos_cits][:5000]
    for cit in distractors:
        corpus[cit] = passage_prefix + citation_to_text[cit]

    print(f"  Val evaluator: {len(queries)} queries, {len(corpus):,} corpus docs")

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="swiss-legal-val",
        show_progress_bar=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mode", choices=["original", "translated", "both"],
                        default="both")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for model output directory. Defaults to 'scenario-{query_mode}'")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-device batch size. Default 4 for P100 16GB.")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable fp16 (use if GPU doesn't support it)")
    args = parser.parse_args()

    query_mode = args.query_mode
    run_name = args.run_name or f"scenario-{query_mode}"
    output_dir = Path(cfg.train.models_root) / run_name
    data_dir = Path(cfg.data.prepared_data_root) / query_mode

    num_epochs = args.epochs or cfg.train.num_epochs
    batch_size = args.batch_size or cfg.train.batch_size

    print(f"\n{'='*60}")
    print(f"  Fine-tuning — mode: {query_mode}  run: {run_name}")
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

    # MultipleNegativesRankingLoss only needs (anchor, positive)
    # If we have hard negatives, they act as additional in-batch negatives naturally
    train_dataset = ds["train"].select_columns(["anchor", "positive"])

    # --- Load model (ShawhinT approach) ---
    print(f"\n[2/4] Loading base model: {cfg.model.base_model}")
    model = SentenceTransformer(cfg.model.base_model)

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
    evaluator = build_val_evaluator(
        val_csv=cfg.data.val_csv,
        laws_csv=cfg.data.laws_csv,
        court_csv=cfg.data.court_csv,
        query_prefix=cfg.model.query_prefix,
        passage_prefix=cfg.model.passage_prefix,
    )

    # --- Effective batch size via gradient accumulation ---
    # MNRL benefits from large effective batch (more in-batch negatives)
    # But actual batch per device must be small to fit in VRAM
    # effective_batch = batch_size × grad_accum_steps
    grad_accum = max(1, 16 // batch_size)  # target effective batch of 16
    print(f"\n  Batch per device: {batch_size}  ×  grad_accum: {grad_accum}"
          f"  =  effective batch: {batch_size * grad_accum}")

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
        metric_for_best_model="swiss-legal-val_cosine_ndcg@10",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=False,       # set True on A100/H100 only
        dataloader_pin_memory=False,
        dataloader_drop_last=True,    # avoid DDP hang on uneven last batch
        report_to="none",
    )

    # --- Train (ShawhinT: SentenceTransformerTrainer) ---
    print("\n[4/4] Training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=ds["eval"].select_columns(["anchor", "positive"]),
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
