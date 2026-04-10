"""
Fine-tune Qwen3-Embedding on Swiss legal retrieval triplets.

Requires data/hard_negative_dataset.json (run hard_negative.py first).
Fine-tuned model is saved to models/qwen-swiss-legal/.
Upload that directory as a Kaggle dataset for offline inference.
"""
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import config
from hard_negative import load_dataset
from data_utils import load_val, load_corpus, parse_citations, build_citation_to_text


def build_evaluator() -> EmbeddingSimilarityEvaluator:
    """
    Proxy evaluator on val.csv: pairs (query, first_gold_text) with score=1.0.
    Not a perfect ranking evaluator but gives a useful training signal.
    """
    val = load_val()
    corpus = load_corpus()
    citation_to_text = build_citation_to_text(corpus)

    s1, s2, scores = [], [], []
    for _, row in val.iterrows():
        for cit in parse_citations(row["gold_citations"])[:1]:
            text = citation_to_text.get(cit)
            if text:
                s1.append(row["query"])
                s2.append(text)
                scores.append(1.0)

    return EmbeddingSimilarityEvaluator(
        sentences1=s1,
        sentences2=s2,
        scores=scores,
        main_similarity="cosine",
        name="val",
    )


def train(
    model_name: str = config.BASE_MODEL,
    dataset_path=config.HARD_NEG_DATASET_PATH,
    output_dir=config.FINETUNED_MODEL_PATH,
):
    print(f"[trainer] loading dataset from {dataset_path} ...")
    raw = load_dataset(dataset_path)
    train_dataset = Dataset.from_dict(raw)
    print(f"[trainer] triplets: {len(train_dataset)}")

    print(f"[trainer] loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = build_evaluator()

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        fp16=config.FP16,
        eval_strategy="steps",   # required when eval_steps is set
        eval_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    print("[trainer] starting training ...")
    trainer.train()

    model.save(str(output_dir))
    print(f"[trainer] fine-tuned model saved → {output_dir}")
    return model


if __name__ == "__main__":
    train()
