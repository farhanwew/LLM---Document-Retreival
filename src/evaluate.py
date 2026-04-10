def precision_recall_f1(gold: list[str], pred: list[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for a single query."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0

    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)

    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1(gold_list: list[list[str]], pred_list: list[list[str]]) -> float:
    """Macro F1 across all queries — matches competition metric."""
    assert len(gold_list) == len(pred_list)
    scores = [precision_recall_f1(g, p)[2] for g, p in zip(gold_list, pred_list)]
    return sum(scores) / len(scores)


def full_report(gold_list: list[list[str]], pred_list: list[list[str]]):
    """Print per-query breakdown: precision, recall, F1."""
    print(f"\n{'query':>6} | {'P':>6} | {'R':>6} | {'F1':>6}")
    print("-" * 35)
    all_p, all_r, all_f1 = [], [], []
    for i, (gold, pred) in enumerate(zip(gold_list, pred_list)):
        p, r, f1 = precision_recall_f1(gold, pred)
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
        print(f"{i:>6} | {p:>6.3f} | {r:>6.3f} | {f1:>6.3f}")
    print("-" * 35)
    print(
        f"{'macro':>6} | {sum(all_p)/len(all_p):>6.3f} | "
        f"{sum(all_r)/len(all_r):>6.3f} | {sum(all_f1)/len(all_f1):>6.3f}"
    )
