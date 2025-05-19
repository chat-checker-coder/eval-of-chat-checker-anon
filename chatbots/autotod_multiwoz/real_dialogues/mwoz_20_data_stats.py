import json
from pathlib import Path
from typing import Dict, List
import statistics

from lexical_diversity import lex_div as ld


def load_data(file_path: str) -> List[Dict]:
    """Load the MultiWOZ dataset from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_statistics(data: dict) -> Dict:
    """Collect various statistics about the dialogues."""
    stats = {
        "num_dialogues": len(data.keys()),
        "turns_per_dialogue": [],
        "user_utterances": [],
        "system_utterances": [],
        "user_utterance_lengths": [],
        "system_utterance_lengths": [],
        "total_utterance_lengths": [],
        "user_utterance_mtlds": [],
        "system_utterance_mtlds": [],
        "user_utterance_terms": [],
        "system_utterance_terms": [],
    }

    for dialogue in data.values():
        turns = dialogue["log"]
        stats["turns_per_dialogue"].append(len(turns))

        for i, turn in enumerate(turns):
            utterance = turn["text"]
            utterance = utterance.strip()

            n_words = len(utterance.split())
            n_terms = len(set(utterance.split()))

            if i % 2 == 0:
                stats["user_utterances"].append(utterance)
                stats["user_utterance_lengths"].append(n_words)
                stats["user_utterance_terms"].append(n_terms)
            else:  # SYSTEM
                stats["system_utterances"].append(utterance)
                stats["system_utterance_lengths"].append(n_words)
                stats["system_utterance_terms"].append(n_terms)

            stats["total_utterance_lengths"].append(n_words)

    return stats


def calculate_metrics(values: List[float]) -> Dict:
    """Calculate statistical metrics for a list of values."""
    if not values:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0, "mean": 0, "std": 0}

    sorted_values = sorted(values)
    q1 = statistics.quantiles(sorted_values, n=4)[0]
    q3 = statistics.quantiles(sorted_values, n=4)[2]

    return {
        "min": min(values),
        "q1": q1,
        "median": statistics.median(values),
        "q3": q3,
        "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
    }


def main():
    # Load the data
    data_path = Path("./mwoz_20_downloaded/data.json")
    data = load_data(data_path)

    # Compute statistics
    stats = collect_statistics(data)

    print(f"Example user utterance: {stats['user_utterances'][0]}")
    print(f"Example system utterance: {stats['system_utterances'][0]}")

    # Calculate and print metrics
    print("\nMultiWOZ Dataset Statistics")
    print("=" * 50)

    print(f"\nNumber of dialogues: {stats['num_dialogues']}")
    print(f"\nNumber of user utterances: {len(stats['user_utterances'])}")
    print(f"\nNumber of system utterances: {len(stats['system_utterances'])}")

    print("\nTurns per dialogue:")
    turns_metrics = calculate_metrics(stats["turns_per_dialogue"])
    for metric, value in turns_metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nTurns per dialogue (without outliers):")
    # Remove outliers using IQR method
    q1, q3 = turns_metrics["q1"], turns_metrics["q3"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    turns_no_outliers = [
        t for t in stats["turns_per_dialogue"] if lower_bound <= t <= upper_bound
    ]
    turns_no_outliers_metrics = calculate_metrics(turns_no_outliers)
    for metric, value in turns_no_outliers_metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nAverage user utterance lengths (words):")
    user_metrics = calculate_metrics(stats["user_utterance_lengths"])
    for metric, value in user_metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nAverage system utterance lengths (words):")
    system_metrics = calculate_metrics(stats["system_utterance_lengths"])
    for metric, value in system_metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nAverage utterance lengths (words):")
    total_metrics = calculate_metrics(stats["total_utterance_lengths"])
    for metric, value in total_metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nAverage user utterance unique terms:")
    user_unique_metrics = calculate_metrics(stats["user_utterance_terms"])
    for metric, value in user_unique_metrics.items():
        print(f"{metric}: {value:.2f}")

    print("\nAverage system utterance unique terms:")
    system_unique_metrics = calculate_metrics(stats["system_utterance_terms"])
    for metric, value in system_unique_metrics.items():
        print(f"{metric}: {value:.2f}")

    all_user_tokens = []
    for utterance in stats["user_utterances"]:
        tokens = ld.tokenize(utterance)
        all_user_tokens.extend(tokens)

    all_system_tokens = []
    for utterance in stats["system_utterances"]:
        tokens = ld.tokenize(utterance)
        all_system_tokens.extend(tokens)

    print("\nUnique user utterance terms:")
    print(len(set(all_user_tokens)))

    print("\nUnique system utterance terms:")
    print(len(set(all_system_tokens)))

    print("\nTotal user utterance MTLD:")
    lex_div_mtld = ld.mtld(all_user_tokens)
    print(f"Total user utterance MTLD (lex_div): {lex_div_mtld}")

    print("\nTotal system utterance MTLD:")
    lex_div_mtld = ld.mtld(all_system_tokens)
    print(f"Total system utterance MTLD (lex_div): {lex_div_mtld}")


if __name__ == "__main__":
    main()
