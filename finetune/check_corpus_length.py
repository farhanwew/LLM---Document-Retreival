"""
Check token length distribution of corpus documents.
Usage: python finetune/check_corpus_length.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from finetune.config import cfg

tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)

results = {}
for label, path in [("laws_de", cfg.data.laws_csv), ("court", cfg.data.court_csv)]:
    df = pd.read_csv(path)[["citation", "text"]].dropna()
    print(f"\n{label}: {len(df):,} docs — sampling 10,000 for speed...")
    sample = df["text"].sample(min(10_000, len(df)), random_state=42).tolist()

    lengths = []
    batch = 512
    for i in range(0, len(sample), batch):
        enc = tokenizer(sample[i:i+batch], truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])

    lengths = np.array(lengths)
    results[label] = lengths

    for threshold in [64, 128, 256, 512]:
        pct = (lengths <= threshold).mean() * 100
        print(f"  <= {threshold:4d} tokens: {pct:5.1f}%")
    print(f"  mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  p95={np.percentile(lengths,95):.1f}  max={lengths.max()}")
