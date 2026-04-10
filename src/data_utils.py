import pandas as pd
from pathlib import Path
import config


def load_train() -> pd.DataFrame:
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"[data] train: {len(df)} rows")
    return df


def load_val() -> pd.DataFrame:
    df = pd.read_csv(config.VAL_PATH)
    print(f"[data] val: {len(df)} rows")
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(config.TEST_PATH)
    print(f"[data] test: {len(df)} rows")
    return df


def load_corpus() -> pd.DataFrame:
    """Load and concatenate laws_de + court_considerations into one corpus."""
    print("[data] loading laws_de.csv ...")
    laws = pd.read_csv(config.LAWS_PATH)

    print("[data] loading court_considerations.csv (big!) ...")
    court = pd.read_csv(config.COURT_PATH)

    corpus = pd.concat([laws, court], ignore_index=True)
    corpus = corpus.dropna(subset=["citation", "text"])
    corpus = corpus.drop_duplicates(subset=["citation"])
    corpus = corpus.reset_index(drop=True)

    print(f"[data] corpus: {len(corpus)} rows total")
    return corpus


def parse_citations(citation_str: str) -> list[str]:
    """Split semicolon-separated citation string into list."""
    if pd.isna(citation_str) or citation_str == "":
        return []
    return [c.strip() for c in citation_str.split(";") if c.strip()]


def build_citation_to_text(corpus: pd.DataFrame) -> dict:
    """Build lookup dict: citation string → text."""
    return dict(zip(corpus["citation"], corpus["text"]))


def eda_summary(train: pd.DataFrame, val: pd.DataFrame, corpus: pd.DataFrame):
    """Print quick EDA."""
    print("\n" + "=" * 50)
    print("EDA SUMMARY")
    print("=" * 50)

    train = train.copy()
    train["n_citations"] = train["gold_citations"].apply(
        lambda x: len(parse_citations(x))
    )
    print(f"\nTrain queries       : {len(train)}")
    print(f"Val queries         : {len(val)}")
    print(f"Corpus size         : {len(corpus)}")
    print(f"\nGold citations per query (train):")
    print(train["n_citations"].describe().to_string())

    citation_set = set(corpus["citation"].tolist())
    train["all_in_corpus"] = train["gold_citations"].apply(
        lambda x: all(c in citation_set for c in parse_citations(x))
    )
    pct = train["all_in_corpus"].mean() * 100
    print(f"\nQueries where ALL gold citations are in corpus: {pct:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train = load_train()
    val = load_val()
    corpus = load_corpus()
    eda_summary(train, val, corpus)
