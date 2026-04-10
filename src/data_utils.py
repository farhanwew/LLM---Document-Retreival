import re
import pandas as pd
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


def parse_citation_year(citation: str) -> int | None:
    """Extract approximate year from a citation string.

    Handles two formats:
      - Non-leading: "5A_800/2019 E 2."  → 2019
      - BGE leading:  "BGE 145 II 32 E."  → 1874 + 145 = 2019 (approx)
    """
    # Non-leading decisions: docket number contains /YYYY
    m = re.search(r"/(\d{4})\b", citation)
    if m:
        return int(m.group(1))

    # BGE leading decisions: volume number encodes year
    m = re.search(r"BGE\s+(\d+)", citation)
    if m:
        volume = int(m.group(1))
        return 1874 + volume  # BGE vol 1 = 1875, approx mapping

    return None


def load_corpus(min_year: int = config.CORPUS_MIN_YEAR) -> pd.DataFrame:
    """Load and concatenate laws_de + court_considerations.

    Court decisions older than min_year are dropped — they are not expected
    in gold citations per competition data description.
    """
    print("[data] loading laws_de.csv ...")
    laws = pd.read_csv(config.LAWS_PATH)

    print("[data] loading court_considerations.csv (big!) ...")
    court = pd.read_csv(config.COURT_PATH)

    # Filter old court decisions
    court["_year"] = court["citation"].apply(parse_citation_year)
    before = len(court)
    court = court[court["_year"].isna() | (court["_year"] >= min_year)]
    court = court.drop(columns=["_year"])
    print(f"[data] court: kept {len(court):,} / {before:,} rows (year >= {min_year})")

    corpus = pd.concat([laws, court], ignore_index=True)
    corpus = corpus.dropna(subset=["citation", "text"])
    corpus = corpus.drop_duplicates(subset=["citation"])
    corpus = corpus.reset_index(drop=True)

    print(f"[data] corpus total: {len(corpus):,} rows")
    return corpus


def parse_citations(citation_str: str) -> list[str]:
    """Split semicolon-separated citation string into list."""
    if pd.isna(citation_str) or citation_str == "":
        return []
    return [c.strip() for c in citation_str.split(";") if c.strip()]


def build_citation_to_text(corpus: pd.DataFrame) -> dict:
    return dict(zip(corpus["citation"], corpus["text"]))


def eda_summary(train: pd.DataFrame, val: pd.DataFrame, corpus: pd.DataFrame):
    print("\n" + "=" * 50)
    print("EDA SUMMARY")
    print("=" * 50)

    train = train.copy()
    train["n_citations"] = train["gold_citations"].apply(
        lambda x: len(parse_citations(x))
    )
    print(f"\nTrain queries : {len(train)}")
    print(f"Val queries   : {len(val)}")
    print(f"Corpus size   : {len(corpus):,}")
    print(f"\nGold citations per query (train):")
    print(train["n_citations"].describe().to_string())

    citation_set = set(corpus["citation"].tolist())
    train["all_in_corpus"] = train["gold_citations"].apply(
        lambda x: all(c in citation_set for c in parse_citations(x))
    )
    pct = train["all_in_corpus"].mean() * 100
    print(f"\nQueries where ALL gold citations in corpus: {pct:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train = load_train()
    val = load_val()
    corpus = load_corpus()
    eda_summary(train, val, corpus)
